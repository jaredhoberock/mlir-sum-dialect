use melior::{
    ir::{r#type::TypeLike, Location, Operation, Type, Value, ValueLike},
    Context,
};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirType, MlirValue};

#[link(name = "sum_dialect")]
unsafe extern "C" {
    fn sumRegisterDialect(ctx: MlirContext);
    fn sumSumTypeCreate(ctx: MlirContext, variants: *const MlirType, nVariants: isize) -> MlirType;
    fn sumGetOpCreate(loc: MlirLocation, input: MlirValue, index: i64) -> MlirOperation;
    fn sumMakeOpCreate(loc: MlirLocation, resultTy: MlirType, index: i64, payload: MlirValue) -> MlirOperation;
    fn sumMatchOpCreate(loc: MlirLocation, input: MlirValue, resultTypes: *const MlirType, nResults: isize) -> MlirOperation;
    fn sumTagOpCreate(loc: MlirLocation, input: MlirValue) -> MlirOperation;
    fn sumYieldOpCreate(loc: MlirLocation, results: *const MlirValue, nResults: isize) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { sumRegisterDialect(context.to_raw()) }
}

pub fn sum_type<'c>(context: &'c Context, variants: &[Type<'c>]) -> Type<'c> {
    unsafe {
        let raw = sumSumTypeCreate(
            context.to_raw(),
            variants.as_ptr() as *const _,
            variants.len() as isize,
        );
        Type::from_raw(raw)
    }
}

pub fn get<'c>(
    loc: Location<'c>,
    input: Value<'c, '_>,
    index: usize,
) -> Operation<'c> {
    unsafe {
        let op = sumGetOpCreate(
            loc.to_raw(),
            input.to_raw(),
            index as i64,
        );
        Operation::from_raw(op)
    }
}

pub fn make<'c>(
    loc: Location<'c>,
    result_ty: Type<'c>,
    index: usize,
    payload: Value<'c, '_>,
) -> Operation<'c> {
    unsafe {
        let op = sumMakeOpCreate(
            loc.to_raw(),
            result_ty.to_raw(),
            index as i64,
            payload.to_raw(),
        );
        Operation::from_raw(op)
    }
}

pub fn match_<'c>(
    loc: Location<'c>,
    input: Value<'c, '_>,
    result_types: &[Type<'c>],
) -> Operation<'c> {
    unsafe {
        let op = sumMatchOpCreate(
            loc.to_raw(),
            input.to_raw(),
            result_types.as_ptr() as *const _,
            result_types.len() as isize,
        );
        Operation::from_raw(op)
    }
}

pub fn tag<'c>(
    loc: Location<'c>,
    input: Value<'c, '_>,
) -> Operation<'c> {
    unsafe {
        let op = sumTagOpCreate(
            loc.to_raw(),
            input.to_raw(),
        );
        Operation::from_raw(op)
    }
}

pub fn yield_<'c>(
    loc: Location<'c>,
    results: &[Value<'c, '_>],
) -> Operation<'c> {
    unsafe {
        let op = sumYieldOpCreate(
            loc.to_raw(),
            results.as_ptr() as *const _,
            results.len() as isize,
        );
        Operation::from_raw(op)
    }
}
