# -*- coding: utf-8 -*-
"""A prototype class to submit processes in batches, avoiding to submit too many."""
import abc
import time

from aiida.engine import submit
from aiida.orm import Group, QueryBuilder, ProcessNode

class BaseSubmissionController:
    """Controller to submit a maximum number of processes (workflows or calculations) at a given time.

    This is an abstract base class: you need to subclass it and define the abstract methods.
    """
    def __init__(self, group_label, max_concurrent):
        """Create a new controller to manage (and limit) concurrent submissions.

        :param group_label: a group label: the group will be created at instantiation (if not existing already,
            and it will be used to manage the calculations)
        :param max_concurrent: maximum number of processes that can run concurrently.
        :param extra_unique_keys: a tuple or list of keys of extras that are used to uniquely identify
            a process in the group. E.g. ('value1', 'value2'). #TODO Seems wrong?

        :note: try to use actual values that allow for an equality comparison (strings, bools, integers), and avoid
           floats, because of truncation errors.
        """
        self._group_label = group_label
        self._max_concurrent = max_concurrent

        # Create the group if needed
        self._group, _ = Group.objects.get_or_create(self.group_label)

    @property
    def group_label(self):
        """Return the label of the group that is managed by this class."""
        return self._group_label

    @property
    def group(self):
        """Return the AiiDA ORM Group instance that is managed by this class."""
        return self._group

    @property
    def max_concurrent(self):
        """Value of the maximum number of concurrent processes that can be run."""
        return self._max_concurrent

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
        return len(self.get_all_extras_to_submit().difference(
            self._check_submitted_extras()))

    @property
    def num_already_run(self):
        """Number of processes that have already been submitted (and might or might not have finished)."""
        return len(self._check_submitted_extras())

    def get_query(self, process_projections, only_active=False):
        """Return a QueryBuilder object to get all processes in the group associated to this.

        Projections on the process must be specified.

        :note: the query has the process already tagged with tag "process", so you can further append to this query
             using this tag, e.g. to get the outputs of the process itself.

        :param process_projections: a list of projections for the ProcessNode.
        :param only_active: if True, will filter only on active (not-sealed) processes.
        """
        qbuild = QueryBuilder()
        filters = {}
        if only_active:
            filters = {
                'or': [{
                    'attributes.sealed': False
                }, {
                    'attributes': {
                        '!has_key': 'sealed'
                    }
                }]
            }

        qbuild.append(Group,
                      filters={'label': self.group_label},
                      tag='group')
        qbuild.append(ProcessNode,
                      project=process_projections,
                      filters=filters,
                      tag='process',
                      with_group='group')
        return qbuild

    def get_process_extra_projections(self):
        """Return a list of QueryBuilder projections on the process.

        They will return the values of the extras according to the output of ``get_extra_unique_keys()``
        (in the same order).

        The idea is to be used as ``process_projections`` for ``get_query()``.
        """
        return [
            f'extras.{unique_key}'
            for unique_key in self.get_extra_unique_keys()
        ]

    def get_all_submitted_pks(self):
        """Return a dictionary of all processes that have been already submitted (i.e. are in the group).

        :return: a dictionary where:

            - the keys are tuples with the values of extras that uniquely identify processes, in the same
              order as returned by get_extra_unique_keys().

            - the values are the corresponding process PKs.

        :note: this returns all processes, both active and completed (sealed).
        """
        projections = self.get_process_extra_projections() + ['id']

        qbuild = self.get_query(only_active=False,
                                process_projections=projections)
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
        projections = self.get_process_extra_projections() + ['*']

        qbuild = self.get_query(only_active=only_active,
                                process_projections=projections)
        all_submitted = {}
        for data in qbuild.all():
            all_submitted[tuple(data[:-1])] = data[-1]

        return all_submitted

    def _check_submitted_extras(self):
        """Return a set with the extras of the processes tha have been already submitted."""
        return set(self.get_all_submitted_pks().keys())

    def _count_active_in_group(self):
        """Count how many active (unsealed) processes there are in the group."""
        qbuild = self.get_query(process_projections=['id'], only_active=True)
        return qbuild.count()

    def submit_new_batch(self, dry_run=False, sort=True, verbose=False, sleep=1):
        """Submit a new batch of calculations, ensuring less than self.max_concurrent active at the same time.
        
        :param dry_run: simply return the extras that would be submitted.
        :param sort: sort the work chains by the extras before submissions.
        :param verbose: print details on submission.
        :param sleep: time to sleep (in seconds) between submissions.
        """

        extras_to_run = list(set(self.get_all_extras_to_submit()).difference(self._check_submitted_extras()))

        if sort is True:
            extras_to_run = sorted(extras_to_run)
        elif sort:
            extras_to_run = sorted(extras_to_run, key=sort)

        to_submit = extras_to_run[:self.num_available_slots]

        if dry_run:
            return {key: None for key in to_submit}

        submitted = {}
        for workchain_extras in to_submit:
            try:
                # Get the inputs and the process calculation for submission
                inputs, process_class = self.get_inputs_and_processclass_from_extras(workchain_extras, dry_run)

                # Actually submit
                workchain_node = submit(process_class, **inputs)

                # Add extras, and put in group
                workchain_node.set_extra_many(dict(zip(self.get_extra_unique_keys(), workchain_extras)))
                self.group.add_nodes([workchain_node])
                submitted[workchain_extras] = workchain_node

                if verbose:
                    print(f'Submitted extras {workchain_extras} -> {workchain_node}')
            except Exception as exception:
                print(f'Failed to submit extras {workchain_extras}: {exception}')

            time.sleep(sleep)

        return submitted

    @abc.abstractmethod
    def get_extra_unique_keys(self):
        """Return a tuple of the keys of the extras that are used to uniquely identify your work chains."""
        return

    @abc.abstractmethod
    def get_all_extras_to_submit(self):
        """Return a *set* of the values of all extras uniquely identifying all work chains that you want to submit.

        Each entry of the set must be a tuple, in same order as the keys returned by get_extra_unique_keys().

        :note: for each item, pass extra values as tuples (because lists are not hashable, so you cannot make
            a set out of them).
        """
        return

    @abc.abstractmethod
    def get_inputs_and_processclass_from_extras(self, extras_values, dry_run):
        """Return the inputs and the process class for the process to run, associated a given tuple of extras values.

        :param extras_values: a tuple of values of the extras, in same order as the keys returned by
            get_extra_unique_keys().

        :return: ``(inputs, process_class)``, that will be used as follows:

           submit(process_class, **inputs)
        """
        return
