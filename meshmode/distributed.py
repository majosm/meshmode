from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2017 Ellis Hoag
Copyright (C) 2017 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np  # noqa

# This file needs to be importable without mpi4py. So don't be tempted to add
# that import here--push it into individual functions instead.

import logging
logger = logging.getLogger(__name__)

TAG_BASE = 83411
TAG_DISTRIBUTE_MESHES = TAG_BASE + 1
TAG_SEND_BOUNDARY = TAG_BASE + 2

__doc__ = """
.. autoclass:: MPIMeshDistributor
.. autoclass:: MPIBoundaryTransceiver

.. autofunction:: get_partition_by_pymetis
.. autofunction:: get_connected_partitions
"""


# {{{ mesh distributor

class MPIMeshDistributor(object):
    """
    .. automethod:: is_mananger_rank
    .. automethod:: send_mesh_parts
    .. automethod:: recv_mesh_part
    """
    def __init__(self, mpi_comm, manager_rank=0):
        """
        :arg mpi_comm: A :class:`MPI.Intracomm`
        """
        self.mpi_comm = mpi_comm
        self.manager_rank = manager_rank

    def is_mananger_rank(self):
        return self.mpi_comm.Get_rank() == self.manager_rank

    def send_mesh_parts(self, mesh, part_per_element, num_parts):
        """
        :arg mesh: A :class:`Mesh` to distribute to other ranks.
        :arg part_per_element: A :class:`numpy.ndarray` containing one
            integer per element of *mesh* indicating which part of the
            partitioned mesh the element is to become a part of.
        :arg num_parts: The number of partitions to divide the mesh into.

        Sends each partition to a different rank.
        Returns one partition that was not sent to any other rank.
        """
        mpi_comm = self.mpi_comm
        rank = mpi_comm.Get_rank()
        assert num_parts <= mpi_comm.Get_size()

        assert self.is_mananger_rank()

        from meshmode.mesh.processing import partition_mesh
        parts = [partition_mesh(mesh, part_per_element, i)[0]
                        for i in range(num_parts)]

        local_part = None

        reqs = []
        for r, part in enumerate(parts):
            if r == self.manager_rank:
                local_part = part
            else:
                reqs.append(mpi_comm.isend(part, dest=r, tag=TAG_DISTRIBUTE_MESHES))

        logger.info('rank %d: sent all mesh partitions', rank)
        for req in reqs:
            req.wait()

        return local_part

    def receive_mesh_part(self):
        """
        Returns the mesh sent by the manager rank.
        """
        mpi_comm = self.mpi_comm
        rank = mpi_comm.Get_rank()

        assert not self.is_mananger_rank(), "Manager rank cannot receive mesh"

        from mpi4py import MPI
        status = MPI.Status()
        result = self.mpi_comm.recv(
                source=self.manager_rank, tag=TAG_DISTRIBUTE_MESHES,
                status=status)
        logger.info('rank %d: received local mesh (size = %d)', rank, status.count)

        return result

# }}}


# {{{ boundary communication setup helper

class MPIBoundaryCommSetupHelper(object):
    """
    .. automethod:: __call__
    .. automethod:: is_ready
    """
    def __init__(self, mpi_comm, queue, local_bdry_conn, i_remote_part,
            bdry_grp_factory):
        """
        :arg mpi_comm: A :class:`MPI.Intracomm`
        :arg queue:
        :arg i_remote_part: The part number of the remote partition
        """
        self.mpi_comm = mpi_comm
        self.queue = queue
        self.i_local_part = mpi_comm.Get_rank()
        self.i_remote_part = i_remote_part
        self.local_bdry_conn = local_bdry_conn
        self.bdry_grp_factory = bdry_grp_factory

    def _post_send_boundary_data(self):
        local_bdry = self.local_bdry_conn.to_discr
        local_mesh = self.local_bdry_conn.from_discr.mesh
        local_adj_groups = [local_mesh.facial_adjacency_groups[i][None]
                            for i in range(len(local_mesh.groups))]
        local_batches = [self.local_bdry_conn.groups[i].batches
                         for i in range(len(local_mesh.groups))]
        local_to_elem_faces = [[batch.to_element_face for batch in grp_batches]
                                for grp_batches in local_batches]
        local_to_elem_indices = [[batch.to_element_indices.get(queue=self.queue)
                                        for batch in grp_batches]
                                    for grp_batches in local_batches]

        local_data = {'bdry_mesh': local_bdry.mesh,
                      'adj': local_adj_groups,
                      'to_elem_faces': local_to_elem_faces,
                      'to_elem_indices': local_to_elem_indices}
        return self.mpi_comm.isend(local_data,
                                   dest=self.i_remote_part,
                                   tag=TAG_SEND_BOUNDARY)

    def post_sends(self):
        logger.info("bdry comm rank %d send begin", self.i_local_part)
        self.send_req = self._post_send_boundary_data()

    def is_setup_ready(self):
        """
        Returns True if the rank boundary data is ready to be received.
        """
        return self.mpi_comm.Iprobe(source=self.i_remote_part, tag=TAG_SEND_BOUNDARY)

    def complete_setup(self):
        """
        Returns the tuple ``remote_to_local_bdry_conn``
        where `remote_to_local_bdry_conn` is a
        :class:`DirectDiscretizationConnection` that gives the connection that
        performs data exchange across faces from partition `i_remote_part` to the
        local mesh.
        """
        remote_data = self.mpi_comm.recv(
                source=self.i_remote_part,
                tag=TAG_SEND_BOUNDARY)

        logger.debug('rank %d: Received rank %d data',
                     self.i_local_part, self.i_remote_part)

        from meshmode.discretization import Discretization
        remote_bdry_mesh = remote_data['bdry_mesh']
        remote_bdry = Discretization(self.queue.context, remote_bdry_mesh,
                                     self.bdry_grp_factory)
        remote_adj_groups = remote_data['adj']
        remote_to_elem_faces = remote_data['to_elem_faces']
        remote_to_elem_indices = remote_data['to_elem_indices']

        # Connect local_mesh to remote_mesh
        from meshmode.discretization.connection import make_partition_connection
        remote_to_local_bdry_conn = make_partition_connection(self.local_bdry_conn,
                                                              self.i_local_part,
                                                              remote_bdry,
                                                              remote_adj_groups,
                                                              remote_to_elem_faces,
                                                              remote_to_elem_indices)
        self.send_req.wait()
        return remote_to_local_bdry_conn

# }}}


def get_partition_by_pymetis(mesh, num_parts, **kwargs):
    """Return a mesh partition created by :mod:`pymetis`.

    :arg mesh: A :class:`meshmode.mesh.Mesh` instance
    :arg num_parts: the number of parts in the mesh partition
    :arg kwargs: Passed unmodified to :func:`pymetis.part_graph`.
    :returns: a :class:`numpy.ndarray` with one entry per element indicating
        to which partition each element belongs, with entries between ``0`` and
        ``num_parts-1``.
    """
    from pymetis import part_graph
    _, p = part_graph(num_parts,
                      xadj=mesh.nodal_adjacency.neighbors_starts.tolist(),
                      adjncy=mesh.nodal_adjacency.neighbors.tolist(),
                      **kwargs)

    return np.array(p)


def get_connected_partitions(mesh):
    """For a local mesh part in *mesh*, determine the set of numbers
    of remote partitions to which this mesh piece is connected.

    :arg mesh: A :class:`meshmode.mesh.Mesh` instance
    :returns: the set of partition numbers that are connected to `mesh`
    """
    connected_parts = set()
    from meshmode.mesh import InterPartitionAdjacencyGroup
    for adj in mesh.facial_adjacency_groups:
        grp = adj.get(None, None)
        if isinstance(grp, InterPartitionAdjacencyGroup):
            indices = grp.neighbor_partitions >= 0
            connected_parts = connected_parts.union(
                    grp.neighbor_partitions[indices])

    return connected_parts


# vim: foldmethod=marker
