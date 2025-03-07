# -*- coding: utf-8 -*-
"""A prototype class to submit processes in batches, avoiding to submit too many."""
import abc
import logging
from typing import Optional

from aiida import engine, orm
from aiida.common import NotExistent
from pydantic import BaseModel, validator
from rich import print
from rich.console import Console
from rich.table import Table

CMDLINE_LOGGER = logging.getLogger("verdi")


def validate_group_exists(value: str) -> str:
    """Validator that makes sure the ``Group`` with the provided label exists."""
    try:
        orm.Group.collection.get(label=value)
    except NotExistent as exc:
        raise ValueError(f"Group with label `{value}` does not exist.") from exc
    else:
        return value


class BaseSubmissionController(BaseModel):
    """Controller to submit a maximum number of processes (workflows or calculations) at a given time.

    This is an abstract base class: you need to subclass it and define the abstract methods.
    """

    group_label: str
    """Label of the group to store the process nodes in."""
    max_concurrent: int
    """Maximum concurrent active processes."""
    unique_extra_keys: Optional[tuple]
    """List of keys defined in the extras that uniquely define each process to be run."""

    _validate_group_exists = validator("group_label", allow_reuse=True)(validate_group_exists)

    @property
    def group(self):
        """Return the AiiDA ORM Group instance that is managed by this class."""
        return orm.Group.objects.get(label=self.group_label)

    def get_query(self, process_projections, only_active=False):
        """Return a QueryBuilder object to get all processes in the group associated to this.

        Projections on the process must be specified.

        :note: the query has the process already tagged with tag "process", so you can further append to this query
             using this tag, e.g. to get the outputs of the process itself.

        :param process_projections: a list of projections for the ProcessNode.
        :param only_active: if True, will filter only on active (not-sealed) processes.
        """
        qbuild = orm.QueryBuilder()
        filters = {}
        if only_active:
            filters = {
                "or": [
                    {"attributes.sealed": False},
                    {"attributes": {"!has_key": "sealed"}},
                ]
            }

        qbuild.append(orm.Group, filters={"label": self.group_label}, tag="group")
        qbuild.append(
            orm.ProcessNode,
            project=process_projections,
            filters=filters,
            tag="process",
            with_group="group",
        )
        return qbuild

    def get_process_extra_projections(self):
        """Return a list of QueryBuilder projections on the process.

        They will return the values of the extras according to the output of ``get_extra_unique_keys()``
        (in the same order).

        The idea is to be used as ``process_projections`` for ``get_query()``.
        """
        return [f"extras.{unique_key}" for unique_key in self.get_extra_unique_keys()]

    def get_all_submitted_pks(self):
        """Return a dictionary of all processes that have been already submitted (i.e., are in the group).

        :return: a dictionary where:

            - the keys are the tuples with the values of extras that uniquely identifies processes, in the same
              order as returned by get_extra_unique_keys().

            - the values are the corresponding process PKs.

        :note: this returns all processes, both active and completed (sealed).
        """
        projections = self.get_process_extra_projections() + ["id"]

        qbuild = self.get_query(only_active=False, process_projections=projections)
        all_submitted = {}
        for data in qbuild.all():
            # Skip nodes without (all of) the right extras
            if any(extra is None for extra in data[:-1]):
                continue
            all_submitted[tuple(data[:-1])] = data[-1]

        return all_submitted

    def get_all_submitted_processes(self, only_active=False):
        """Return a dictionary of all processes that have been already submitted (i.e., are in the group).

        :return: a dictionary where:

            - the keys are the tuples with the values of extras that uniquely identifies processes, in the same
              order as returned by get_extra_unique_keys().

            - the values are the corresponding AiiDA ProcessNode instances.

        :note: this returns all processes, both active and completed (sealed).
        """
        projections = self.get_process_extra_projections() + ["*"]

        qbuild = self.get_query(only_active=only_active, process_projections=projections)
        all_submitted = {}
        for data in qbuild.all():
            all_submitted[tuple(data[:-1])] = data[-1]

        return all_submitted

    def _check_submitted_extras(self):
        """Return a set with the extras of the processes tha have been already submitted."""
        return set(self.get_all_submitted_pks().keys())

    def _count_active_in_group(self):
        """Count how many active (unsealed) processes there are in the group."""
        qbuild = self.get_query(process_projections=["id"], only_active=True)
        return qbuild.count()

    @property
    def num_active_slots(self):
        """Number of active slots (i.e. processes in the group that are unsealed)."""
        return self._count_active_in_group()

    @property
    def num_available_slots(self):
        """Number of available slots (i.e. how many processes would be submitted in the next batch submission)."""
        return max(0, self.max_concurrent - self.num_active_slots)

    @property
    def num_to_run(self):
        """Number of processes that still have to be submitted."""
        return len(self.get_all_extras_to_submit().difference(self._check_submitted_extras()))

    @property
    def num_already_run(self):
        """Number of processes that have already been submitted (and might or might not have finished)."""
        return len(self._check_submitted_extras())

    def submit_new_batch(self, dry_run=False, sort=True, verbose=False):
        """Submit a new batch of calculations, ensuring less than self.max_concurrent active at the same time."""
        CMDLINE_LOGGER.level = logging.INFO if verbose else logging.WARNING
        to_submit = []
        extras_to_run = set(self.get_all_extras_to_submit()).difference(self._check_submitted_extras())
        if sort:
            extras_to_run = sorted(extras_to_run)
        for workchain_extras in extras_to_run:
            if len(to_submit) + self._count_active_in_group() >= self.max_concurrent:
                break
            to_submit.append(workchain_extras)

        if dry_run:
            return {key: None for key in to_submit}

        if verbose:
            table = Table(title="Status")

            table.add_column("Total", justify="left", style="cyan", no_wrap=True)
            table.add_column("Finished", justify="left", style="cyan", no_wrap=True)
            table.add_column("Left to run", justify="left", style="cyan", no_wrap=True)
            table.add_column("Max active", justify="left", style="cyan", no_wrap=True)
            table.add_column("Active", justify="left", style="cyan", no_wrap=True)
            table.add_column("Available", justify="left", style="cyan", no_wrap=True)

            table.add_row(
                str(self.parent_group.count()),
                str(self.num_already_run),
                str(self.num_to_run),
                str(self.max_concurrent),
                str(self.num_active_slots),
                str(self.num_available_slots),
            )
            console = Console()
            console.print(table)

            if len(to_submit) == 0:
                print("[bold blue]Info:[/] 😴 Nothing to submit.")
            else:
                print(f"[bold blue]Info:[/] 🚀 Submitting {len(to_submit)} new workchains!")

        submitted = {}
        for workchain_extras in to_submit:
            try:
                # Get the inputs and the process calculation for submission
                builder = self.get_inputs_and_processclass_from_extras(workchain_extras)

                # Actually submit
                wc_node = engine.submit(builder)

            except (ValueError, TypeError, AttributeError) as exc:
                CMDLINE_LOGGER.error(f"Failed to submit work chain for extras <{workchain_extras}>: {exc}")
            else:
                CMDLINE_LOGGER.report(f"Submitted work chain <{wc_node}> for extras <{workchain_extras}>.")
                # Add extras, and put in group
                wc_node.set_extra_many(dict(zip(self.get_extra_unique_keys(), workchain_extras)))
                self.group.add_nodes([wc_node])
                submitted[workchain_extras] = wc_node

        return submitted

    def get_extra_unique_keys(self):
        """Return a tuple of the kes of the unique extras that will be used to uniquely identify your workchains."""
        return self.unique_extra_keys

    @abc.abstractmethod
    def get_all_extras_to_submit(self):
        """Return a *set* of the values of all extras uniquely identifying all simulations that you want to submit.

        Each entry of the set must be a tuple, in same order as the keys returned by get_extra_unique_keys().

        :note: for each item, pass extra values as tuples (because lists are not hashable, so you cannot make
            a set out of them).
        """
        return

    @abc.abstractmethod
    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return the inputs and the process class for the process to run, associated a given tuple of extras values.

        :param extras_values: a tuple of values of the extras, in same order as the keys returned by
            get_extra_unique_keys().

        :return: ``(inputs, process_class)``, that will be used as follows:

           submit(process_class, **inputs)
        """
        return
