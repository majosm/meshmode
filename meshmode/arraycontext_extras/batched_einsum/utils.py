__copyright__ = """
Copyright (C) 2023 Kaushik Kulkarni
Copyright (C) 2022 Andreas Kloeckner
Copyright (C) 2022 Matthias Diener
Copyright (C) 2022 Matt Smith
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

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Mapping, Set, Tuple, Type)

from immutables import Map

import loopy as lp
import loopy.match as lp_match
import loopy.symbolic as lp_symbolic
import pymbolic.primitives as prim
import pytato as pt
from loopy.translation_unit import for_each_kernel
from pytools import memoize_on_first_arg
from pytools.tag import Tag, ToTagSetConvertible


if TYPE_CHECKING:
    import feinsum

    import pyopencl


# {{{ IndirectionMapsCollector

class IndirectionMapsCollector(pt.transform.CachedWalkMapper):
    """
    .. note::

        We deliberately avoid using :class:`pytato.transform.CombineMapper` since
        the mapper's caching structure would still lead to recomputing
        the union of sets for the results of a revisited node.
    """
    def __init__(self) -> None:
        self.collected_indirection_maps: Set[pt.Array] = set()
        super().__init__()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self,  # type: ignore[override]
                      expr: pt.transform.ArrayOrNames) -> int:
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        if isinstance(expr, pt.IndexBase):
            for idx in expr.indices:
                if isinstance(idx, pt.Array):
                    self.collected_indirection_maps.add(idx)


def get_indirection_maps(expr: pt.DictOfNamedArrays) -> FrozenSet[pt.Array]:
    mapper = IndirectionMapsCollector()
    mapper(expr)
    return frozenset(mapper.collected_indirection_maps)

# }}}


# {{{ EinsumInputOutputCollector

class ReductionInputOutputCollector(pt.transform.CachedWalkMapper):
    """
    .. note::

        We deliberately avoid using :class:`pytato.transform.CombineMapper` since
        the mapper's caching structure would still lead to recomputing
        the union of sets for the results of a revisited node.
    """
    def __init__(self) -> None:
        self.collected_outputs: Set[pt.Array] = set()
        self.collected_inputs: Set[pt.Array] = set()
        super().__init__()

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def get_cache_key(self,  # type: ignore[override]
                      expr: pt.transform.ArrayOrNames) -> int:
        return id(expr)

    # type-ignore-reason: dropped the extra `*args, **kwargs`.
    def post_visit(self, expr: Any) -> None:  # type: ignore[override]
        if isinstance(expr, pt.IndexLambda) and expr.var_to_reduction_descr:
            self.collected_outputs.add(expr)
            self.collected_inputs.update(expr.bindings.values())


def get_inputs_and_outputs_of_reduction_nodes(
        expr: pt.DictOfNamedArrays) -> Tuple[FrozenSet[pt.Array],
                                             FrozenSet[pt.Array]]:
    mapper = ReductionInputOutputCollector()
    mapper(expr)
    return frozenset(mapper.collected_inputs), frozenset(mapper.collected_outputs)

# }}}


def _make_passthrough_arg(ary: pt.Array,
                          tags: ToTagSetConvertible = frozenset()) -> pt.Array:
    from pytato.array import make_index_lambda
    return make_index_lambda(
        prim.Variable("in")[tuple(prim.Variable(f"_{idim}")
                                  for idim in range(ary.ndim))],
        bindings={"in": ary},
        shape=ary.shape,
        dtype=ary.dtype,
    ).tagged(tags)


@dataclass(frozen=True)
class EinsumWithAxesTagged:
    einsum: "feinsum.FusedEinsum"
    index_tags: Mapping["feinsum.EinsumAxisAccess", Tag]

    def __post_init__(self):
        assert (frozenset(self.einsum.index_to_dim_length())
                == frozenset(self.index_tags))


def get_n_callable_kernels(t_unit: lp.TranslationUnit) -> int:
    from loopy.kernel.function_interface import CallableKernel
    return len([name
                for name, clbl in t_unit.callables_table.items()
                if isinstance(clbl, CallableKernel)])


def _get_fusion_order_key(x: "feinsum.EinsumAxisAccess") -> int:
    # Fuse outer loops before reduction loops
    import feinsum as fnsm
    if isinstance(x, fnsm.FreeAxis):
        return 0
    elif isinstance(x, fnsm. SummationAxis):
        return 1
    else:
        raise NotImplementedError(type(x))


def apply_kennedy_fusion_with_batched_einsum_extension(
        t_unit: lp.TranslationUnit,
        tag_t: Type[Tag],
        fused_loop_name_prefix_getter: Callable[[Tag], str]) -> lp.TranslationUnit:

    import feinsum as fnsm

    if get_n_callable_kernels(t_unit) > 1:
        # We accept 't_unit' (instead of a kernel) to comply with feinsum's API.
        raise NotImplementedError(
            "'apply_kennedy_fusion_with_batched_einsum_extension'"
            " does not support multiple callee kernels (yet).")

    kernel = t_unit.default_entrypoint

    assert all(len(kernel.iname_to_insns()[iname]) <= 1
               for iname in kernel.all_inames())

    # A bucket by a tagged einsum and it's position in the einsum.
    bucket_to_inames: Dict[Tuple[EinsumWithAxesTagged, fnsm.EinsumAxisAccess],
                           Set[str]] = {}

    import time

    ts = time.time()

    for insn in kernel.instructions:
        if isinstance(insn, lp.Assignment):
            # {{{ get matching einsum/subst_map

            if insn.reduction_inames():
                einsum, _ = fnsm.get_a_matched_einsum(
                    t_unit, insn_match=lp_match.Id(insn.id))
                einsum = fnsm.canonicalize_einsum(einsum)
                subst_map = fnsm.match_t_unit_to_einsum(
                    t_unit, einsum, insn_match=lp_match.Id(insn.id))
            else:
                # we treat any non-reduction einsum as a copy-einsum
                assignee = insn.assignee
                if isinstance(assignee, prim.Variable):
                    lpy_dim_names = []
                else:
                    assert isinstance(assignee, prim.Subscript)
                    lpy_dim_names = [index.name for index in assignee.index_tuple]

                dim_lengths = [kernel.get_constant_iname_length(dim_name)
                               for dim_name in lpy_dim_names]
                if len(lpy_dim_names) > 26:
                    raise ValueError("Batched Einsum Actx supports upto"
                                     "26-dimensions")
                einsum_dim_names = [chr(97+idim)
                                    for idim in range(len(lpy_dim_names))]
                einsum = fnsm.einsum(
                    f"{''.join(einsum_dim_names)}->{''.join(einsum_dim_names)}",
                    # purposefully fix dtype=F64, since for such expression we are
                    # imprecise on purpose.
                    fnsm.array(shape=dim_lengths, dtype="float64"),
                )
                einsum = fnsm.canonicalize_einsum(einsum)
                subst_map = {
                    einsum.index_names[fnsm.FreeAxis(idim)]: lpy_dim_names[idim]
                    for idim in range(len(einsum_dim_names))}
            # }}}

            idx_tags: Dict[fnsm.EinsumAxisAccess, Tag] = {}
            for acc_descr, name_in_einsum in einsum.index_names.items():
                lpy_iname = subst_map[name_in_einsum]
                lpy_iname_tag, = kernel.iname_tags_of_type(lpy_iname, tag_t)
                idx_tags[acc_descr] = lpy_iname_tag

            tagged_einsum = EinsumWithAxesTagged(einsum, Map(idx_tags))

            for acc_descr, name_in_einsum in einsum.index_names.items():
                bucket = (tagged_einsum, acc_descr)
                lpy_iname = subst_map[name_in_einsum]
                bucket_to_inames.setdefault(bucket, set()).add(lpy_iname)

        else:
            # TODO: should this be a ValueError?
            raise NotImplementedError

    te = time.time()
    print(f"apply_kennedy_fusion_with_..., loop 1: {te-ts}")

    ts = time.time()

    sorted_bucket_to_inames = sorted(
        bucket_to_inames.items(),
        key=lambda x: (_get_fusion_order_key(x[0][1]), sorted(x[1])))

    te = time.time()
    print(f"apply_kennedy_fusion_with_..., sorted_bucket_to_inames: {te-ts}")

    ts = time.time()

    for _, inames in sorted_bucket_to_inames:
        inames_tag, = kernel.iname_tags_of_type(next(iter(inames)),
                                                tag_t)
        # TODO: Enable pylint once these routines have been upstreamed to loopy
        kernel = lp.rename_inames_in_batch(  # pylint: disable = no-member
            kernel,
            lp.get_kennedy_unweighted_fusion_candidates(  # pylint: disable=no-member
                kernel, inames,
                prefix=fused_loop_name_prefix_getter(inames_tag),
            ),
        )

    te = time.time()
    print(f"apply_kennedy_fusion_with_..., loop 2: {te-ts}")

    return t_unit.with_kernel(kernel)


@for_each_kernel
def remove_dead_temporaries(kernel: lp.LoopKernel) -> lp.LoopKernel:
    wmap = kernel.writer_map()
    rmap = kernel.reader_map()

    new_tvs: Dict[str, lp.TemporaryVariable] = {}

    for name, tv in kernel.temporary_variables.items():
        writer_ids: FrozenSet[str] = wmap.get(name, frozenset())
        reader_ids: FrozenSet[str] = rmap.get(name, frozenset())

        if len(writer_ids) != 0 or len(reader_ids) != 0:
            new_tvs[name] = tv

    return kernel.copy(temporary_variables=new_tvs)


class IndexingTupleCollector(lp_symbolic.WalkMapper):
    def __init__(self, subscript_name: str) -> None:
        self.subscript_name = subscript_name
        super().__init__()

        # mutable state:
        self.collected_index_tuples: Set[Tuple[prim.Expression, ...]] = set()

    def post_visit(self, expr: prim.Expression) -> None:
        if (isinstance(expr, prim.Subscript)
                and expr.aggregate == prim.Variable(self.subscript_name)):
            self.collected_index_tuples.add(expr.index_tuple)


@memoize_on_first_arg
def _expand_substs(kernel):
    # memoized wrapper for lp.expand_substs
    return lp.expand_subst(kernel)


@memoize_on_first_arg
def can_temp_var_be_contracted(kernel: lp.LoopKernel, name: str) -> bool:
    kernel = _expand_substs(kernel)
    wmap = kernel.writer_map()
    rmap = kernel.reader_map()

    writer_ids: FrozenSet[str] = wmap.get(name, frozenset())
    reader_ids: FrozenSet[str] = rmap.get(name, frozenset())

    if kernel.temporary_variables[name].initializer:
        # this is a constant literal => cannot be contracted
        return False

    if len(writer_ids) == 0:
        assert len(reader_ids) == 0
        return True
    else:
        mapper = IndexingTupleCollector(name)
        for insn_id in writer_ids | reader_ids:
            insn = kernel.id_to_insn[insn_id]
            mapper((insn.expression,
                    insn.assignees,
                    tuple(insn.predicates)))

        return len(mapper.collected_index_tuples) == 1


class ArrayContracter(lp_symbolic.RuleAwareIdentityMapper):
    def __init__(self,
                 rule_mapping_context: lp_symbolic.SubstitutionRuleMappingContext,
                 arrays_to_contract: FrozenSet[str]):
        self.arrays_to_contract = arrays_to_contract
        super().__init__(rule_mapping_context)

    def map_subscript(self, expr, expn_state) -> prim.Expression:
        if (isinstance(expr.aggregate, prim.Variable)
                and expr.aggregate.name in self.arrays_to_contract):
            return expr.aggregate
        else:
            return super().map_subscript(expr, expn_state)


@for_each_kernel
def contract_arrays(kernel: lp.LoopKernel):
    # Note: We could have used lp.precompute here, but that would be unnecessarily
    # expensive.
    new_tvs: Dict[str, lp.TemporaryVariable] = {}

    rule_mapping_context = lp_symbolic.SubstitutionRuleMappingContext(
        kernel.substitutions,
        kernel.get_var_name_generator()
    )
    temps_to_contract = frozenset({
        name for name, tv in kernel.temporary_variables.items()
        if can_temp_var_be_contracted(kernel, name)})
    array_contracter = ArrayContracter(rule_mapping_context,
                                       temps_to_contract)

    kernel = rule_mapping_context.finish_kernel(
        array_contracter.map_kernel(
            kernel, map_tvs=False, map_args=False)
    )

    for name, tv in kernel.temporary_variables.items():
        if name in temps_to_contract:
            tv = tv.copy(shape=(),
                         dim_tags=(),
                         address_space=lp.AddressSpace.PRIVATE)

        new_tvs[name] = tv

    return kernel.copy(temporary_variables=new_tvs)


@for_each_kernel
def combine_domains_of_perfect_loop_nests(kernel: lp.LoopKernel) -> lp.LoopKernel:
    import islpy as isl

    from meshmode.arraycontext_extras.split_actx.utils import _is_a_perfect_loop_nest

    new_domains: List[isl.BasicSet] = []

    seen_loop_nests: Set[FrozenSet[str]] = set()

    for insn in kernel.instructions:
        assert _is_a_perfect_loop_nest(kernel, insn.within_inames)
        loop_nest = insn.within_inames | insn.reduction_inames()

        if loop_nest in seen_loop_nests:
            continue

        domain = kernel.get_inames_domain(loop_nest)
        new_domains.append(domain.project_out_except(sorted(loop_nest),
                                                     [isl.dim_type.set]))
        seen_loop_nests.add(loop_nest)

    return kernel.copy(domains=new_domains)


def _apply_feinsum_transformations_to_single_kernel(
        t_unit: lp.TranslationUnit, kernel_name: str, feinsum_db: str,
        cl_device: "pyopencl.Device",
) -> lp.TranslationUnit:
    import feinsum as fnsm

    from meshmode.arraycontext_extras.split_actx.utils import (
        InsnIds, _get_call_kernel_insn_ids, _LoopNest,
        _split_loop_nest_across_work_items, get_iname_length)
    call_kernel_insn_ids = _get_call_kernel_insn_ids(t_unit[kernel_name])
    iname_to_length = {iname: get_iname_length(t_unit[kernel_name], iname)
                       for iname in t_unit[kernel_name].all_inames()}

    for insn_ids in call_kernel_insn_ids:
        within_inames, = {t_unit[kernel_name].id_to_insn[insn_id].within_inames
                          for insn_id in insn_ids}
        redn_inames, = {t_unit[kernel_name].id_to_insn[insn_id].reduction_inames()
                        for insn_id in insn_ids}
        if redn_inames:
            einsum, _ = fnsm.get_a_matched_einsum(t_unit,
                                                  insn_match=InsnIds(insn_ids),
                                                  kernel_name=kernel_name,
                                                  long_dim_length=128,
                                                  )
            available_facts = fnsm.query(
                einsum,
                fnsm.make_fake_cl_context([cl_device.name]),
                database=feinsum_db)
            if available_facts:
                best_query = max(
                    available_facts,
                    key=lambda q: sum(q.giga_op_info.values())/q.runtime_in_sec)
                # type-ignore reason: mypy is right here, the callable returned
                # by feinsum is imprecisely typed.
                t_unit = best_query.transform(
                    t_unit,
                    insn_match=InsnIds(insn_ids),  # type: ignore[call-arg]
                    kernel_name=kernel_name)
            else:
                from arraycontext.impl.pytato.compile import FromArrayContextCompile
                if t_unit.default_entrypoint.tags_of_type(FromArrayContextCompile):
                    from warnings import warn
                    warn(f"The database at '{feinsum_db}' has no tuned instances"
                         f" for {einsum}")
                t_unit = t_unit.with_kernel(
                    _split_loop_nest_across_work_items(t_unit[kernel_name],
                                                       _LoopNest(
                                                           within_inames,
                                                           insn_ids),
                                                       iname_to_length,
                                                       cl_device))
        else:
            # TODO: read the grid/block size from the database.
            t_unit = t_unit.with_kernel(
                _split_loop_nest_across_work_items(t_unit[kernel_name],
                                                   _LoopNest(
                                                       within_inames,
                                                       insn_ids),
                                                   iname_to_length,
                                                   cl_device))

    return t_unit


def apply_feinsum_transformations(t_unit: lp.TranslationUnit,
                                  feinsum_db: str,
                                  cl_device: "pyopencl.Device"
                                  ) -> lp.TranslationUnit:
    from loopy.kernel.function_interface import CallableKernel
    kernel_names = {name
                    for name, clbl in t_unit.callables_table.items()
                    if isinstance(clbl, CallableKernel)}
    for kernel_name in kernel_names:
        t_unit = _apply_feinsum_transformations_to_single_kernel(
            t_unit, kernel_name, feinsum_db, cl_device)
    return t_unit

# vim: fdm=marker
