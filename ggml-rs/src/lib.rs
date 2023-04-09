//! # ggml-rs
//! ggml-rs is a Rust implementation of the [ggml](https://github.com/ggerganov/ggml), a tensor library for machine learning used by llama.cpp and whisper.cpp.

use std::{
    cmp::min,
    marker::PhantomData,
    mem::size_of,
    sync::{Arc, Mutex},
};

use num_traits::Num;

const MAX_DIMS: usize = 4;

#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sqr,
    Sqrt,
    Sum,
    Mean,
    Repeat,
    Abs,
    Sgn,
    Neg,
    Step,
    Relu,
    Gelu,
    Silu,
    Norm, // normalize
    RmsNorm,

    MulMat,

    Scale,
    Cpy,
    Reshape,
    View,
    Permute,
    Transpose,
    GetRows,
    DiagMaskInf,
    SoftMax,
    Rope,
    Conv1D1S,
    Conv1D2S,

    FlashAttn,
    FlashFF,

    Count,
}

impl Op {
    fn is_unary_op(&self) -> bool {
        match self {
            Op::Sqr
            | Op::Sqrt
            | Op::Abs
            | Op::Sgn
            | Op::Neg
            | Op::Step
            | Op::Relu
            | Op::Gelu
            | Op::Silu
            | Op::Norm
            | Op::RmsNorm => true,
            _ => false,
        }
    }

    fn is_binary_op(&self) -> bool {
        match self {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::MulMat => true,
            _ => false,
        }
    }
}

// Tensor represents n-dimensional tensor
#[derive(Debug, Clone)]
pub struct Tensor<T: Num + Copy + From<usize>> {
    r#type: PhantomData<T>,
    n_dims: usize,

    ne: [usize; MAX_DIMS],
    nb: [usize; MAX_DIMS],

    op: Option<Op>,

    #[allow(dead_code)]
    is_param: bool,

    grad: Option<Arc<Tensor<T>>>,
    src0: Option<Arc<Tensor<T>>>,
    src1: Option<Arc<Tensor<T>>>,
    // TODO: Add opt[MAX_OPT]
    data: Arc<Mutex<Vec<T>>>,
}

impl<T: Num + Copy + From<usize>> Tensor<T> {
    pub fn new(n_dims: usize, _ne: &[usize], data: Option<Arc<Mutex<Vec<T>>>>) -> Self {
        assert!(n_dims <= MAX_DIMS);
        let mut ne = [1; MAX_DIMS];

        for i in 0..n_dims {
            ne[i] = _ne[i];
        }

        let mut size_needed = 0;
        size_needed += size_of::<T>();
        for i in 0..n_dims {
            size_needed *= ne[i];
        }
        // TODO: align size_needed

        let data = match data {
            Some(data) => data,
            None => Arc::new(Mutex::new(Vec::with_capacity(size_needed))),
        };

        Self {
            r#type: PhantomData,
            n_dims,
            ne,
            nb: [0; MAX_DIMS],
            op: None,
            is_param: false,
            grad: None,
            src0: None,
            src1: None,
            data,
        }
    }

    fn set(&mut self, x: T) {
        let n = self.nrows();
        let nc = self.ne[0];
        let n1 = self.nb[1];

        let mut data = self.data.lock().unwrap();

        for i in 0..n {
            for j in 0..nc {
                data[i * n1 + j] = x;
            }
        }
    }

    pub fn nelements(&self) -> usize {
        self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3]
    }

    pub fn nrows(&self) -> usize {
        self.ne[1] * self.ne[2] * self.ne[3]
    }

    #[inline(always)]
    fn is_scalar(&self) -> bool {
        self.ne[0] == 1 && self.ne[1] == 1 && self.ne[2] == 1 && self.ne[3] == 1
    }

    #[inline(always)]
    fn is_vector(&self) -> bool {
        self.ne[1] == 1 && self.ne[2] == 1 && self.ne[3] == 1
    }

    #[inline(always)]
    fn is_matrix(&self) -> bool {
        self.ne[2] == 1 && self.ne[3] == 1
    }

    #[inline(always)]
    fn is_same_shape(&self, other: &Self) -> bool {
        self.ne == other.ne
    }

    #[inline(always)]
    fn can_repeat(&self, other: &Self) -> bool {
        self.ne[0] % other.ne[0] == 0
            && self.ne[1] % other.ne[1] == 0
            && self.ne[2] % other.ne[2] == 0
            && self.ne[3] % other.ne[3] == 0
    }

    #[inline(always)]
    fn can_mul_mat(&self, other: &Self) -> bool {
        self.ne[0] == other.ne[0] && self.ne[2] == other.ne[2] && self.ne[3] == other.ne[3]
    }

    #[inline(always)]
    fn is_transposed(&self) -> bool {
        self.nb[0] > self.nb[1]
    }

    #[inline(always)]
    fn is_contiguous(&self) -> bool {
        self.nb[0] == size_of::<T>()
            && self.nb[1] == self.nb[0] * self.ne[0]
            && self.nb[2] == self.nb[1] * self.ne[1]
            && self.nb[3] == self.nb[2] * self.ne[2]
    }

    #[inline(always)]
    fn is_padded_1d(&self) -> bool {
        self.nb[0] == size_of::<T>()
            && self.nb[2] == self.nb[1] * self.ne[1]
            && self.nb[3] == self.nb[2] * self.ne[2]
    }
}

pub struct Context<T: Num + Copy + From<usize>> {
    r#type: PhantomData<T>,
}

impl<T: Num + Copy + From<usize>> Context<T> {
    pub const fn new() -> Self {
        Self {
            r#type: PhantomData,
        }
    }

    pub fn init(&mut self, _mem_size: usize) {
        // TODO: Allocate heap memory
    }

    fn new_tensor_with_data(
        &mut self,
        n_dims: usize,
        ne: &[usize],
        data: Arc<Mutex<Vec<T>>>,
    ) -> Tensor<T> {
        Tensor::new(n_dims, ne, Some(data))
    }

    fn new_tensor(&mut self, n_dims: usize, ne: &[usize]) -> Tensor<T> {
        Tensor::new(n_dims, ne, None)
    }

    pub fn new_tensor_1d(&mut self, ne0: usize) -> Tensor<T> {
        let ne = [ne0];
        self.new_tensor(ne.len(), &ne)
    }

    pub fn new_tensor_2d(&mut self, ne0: usize, ne1: usize) -> Tensor<T> {
        let ne = [ne0, ne1];
        self.new_tensor(ne.len(), &ne)
    }

    pub fn new_tensor_3d(&mut self, ne0: usize, ne1: usize, ne2: usize) -> Tensor<T> {
        let ne = [ne0, ne1, ne2];
        self.new_tensor(ne.len(), &ne)
    }

    pub fn new_tensor_4d(&mut self, ne0: usize, ne1: usize, ne2: usize, ne3: usize) -> Tensor<T> {
        let ne = [ne0, ne1, ne2, ne3];
        self.new_tensor(ne.len(), &ne)
    }

    pub fn new_scalar(&mut self, x: T) -> Tensor<T> {
        let mut tensor = self.new_tensor_1d(1);
        tensor.set(x);
        tensor
    }

    pub fn dup_tensor(&mut self, a: &Tensor<T>) -> Tensor<T> {
        self.new_tensor(a.n_dims, &a.ne)
    }

    pub fn view_tensor(&mut self, a: &Tensor<T>) -> Tensor<T> {
        let mut tensor = self.dup_tensor(a);
        tensor.nb = [a.nb[0], a.nb[1], a.nb[2], a.nb[3]];
        tensor
    }

    pub fn op_get_rows(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        assert!(a.is_matrix() && b.is_vector());

        let mut c = self.new_tensor_2d(a.ne[0], b.ne[0]);
        c.op = Some(Op::GetRows);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    fn op_unary_op(&mut self, op: Op, a: Arc<Tensor<T>>, inplace: bool) -> Tensor<T> {
        assert!(op.is_unary_op());

        let mut c = if inplace {
            self.view_tensor(a.as_ref())
        } else {
            self.dup_tensor(a.as_ref())
        };
        c.op = Some(op);
        c.src0 = Some(a.clone());
        if !inplace && a.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    fn op_binary_op(
        &mut self,
        op: Op,
        a: Arc<Tensor<T>>,
        b: Arc<Tensor<T>>,
        inplace: bool,
    ) -> Tensor<T> {
        assert!(op.is_binary_op());
        assert!(a.is_same_shape(&b));

        let mut c = if inplace {
            self.view_tensor(a.as_ref())
        } else {
            self.dup_tensor(a.as_ref())
        };
        c.op = Some(op);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if !inplace && (a.grad.is_some() || b.grad.is_some()) {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_add(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_binary_op(Op::Add, a, b, false)
    }

    pub fn op_sub(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_binary_op(Op::Sub, a, b, false)
    }

    pub fn op_mul(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_binary_op(Op::Mul, a, b, false)
    }

    pub fn op_div(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_binary_op(Op::Div, a, b, false)
    }

    pub fn op_norm(&mut self, a: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_unary_op(Op::Norm, a, false)
    }

    pub fn op_rms_norm(&mut self, a: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_unary_op(Op::RmsNorm, a, false)
    }

    pub fn op_silu(&mut self, a: Arc<Tensor<T>>) -> Tensor<T> {
        self.op_unary_op(Op::Silu, a, false)
    }

    pub fn op_repeat(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        assert!(a.can_repeat(&b));

        let is_node = if a.grad.is_some() { true } else { false };

        if a.is_same_shape(&b) && !is_node {
            // TODO: return a.clone()
        }

        let mut c = self.new_tensor(b.n_dims, &b.ne);
        c.op = Some(Op::Repeat);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if is_node {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_mul_mat(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        assert!(a.can_mul_mat(&b));
        assert!(a.is_transposed() == false);

        let n_dims = min(a.n_dims, b.n_dims);
        let ne = [a.ne[1], b.ne[1], a.ne[2], b.ne[3]];
        let mut c = self.new_tensor(n_dims, &ne);
        c.op = Some(Op::MulMat);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_scale(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        assert!(b.is_scalar());
        assert!(a.is_padded_1d());

        let mut c = self.view_tensor(a.as_ref());
        c.op = Some(Op::Scale);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_diag_mask_inf(&mut self, a: Arc<Tensor<T>>, n_past: usize) -> Tensor<T> {
        let b = Arc::new(self.new_scalar(n_past.into()));

        let mut c = self.view_tensor(a.as_ref());
        c.op = Some(Op::DiagMaskInf);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_soft_max(&mut self, a: Arc<Tensor<T>>) -> Tensor<T> {
        let mut c = self.view_tensor(a.as_ref());
        c.op = Some(Op::SoftMax);
        c.src0 = Some(a.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_view_1d(&mut self, a: Arc<Tensor<T>>, ne0: usize, offset: usize) -> Tensor<T> {
        assert!(a.grad.is_none());

        let mut c = self.new_tensor_1d(ne0);
        c.op = Some(Op::View);
        c.src0 = Some(a.clone());
        c.src1 = Some(Arc::new(self.new_scalar(offset.into())));
        c
    }

    pub fn op_view_2d(
        &mut self,
        a: Arc<Tensor<T>>,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
    ) -> Tensor<T> {
        assert!(a.grad.is_none());

        let mut c = self.new_tensor_2d(ne0, ne1);
        c.nb[1] = nb1;
        c.nb[2] = c.nb[1] * ne1;
        c.nb[3] = c.nb[2];

        c.op = Some(Op::View);
        c.src0 = Some(a.clone());
        c.src1 = Some(Arc::new(self.new_scalar(offset.into())));
        c
    }

    pub fn op_view_3d(
        &mut self,
        a: Arc<Tensor<T>>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> Tensor<T> {
        assert!(a.grad.is_none());

        let mut c = self.new_tensor_3d(ne0, ne1, ne2);
        c.nb[1] = nb1;
        c.nb[2] = nb2;
        c.nb[3] = c.nb[2] * ne2;

        c.op = Some(Op::View);
        c.src0 = Some(a.clone());
        c.src1 = Some(Arc::new(self.new_scalar(offset.into())));
        c
    }

    pub fn op_cpy(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        assert!(a.nelements() == b.nelements());

        let mut c = self.view_tensor(b.as_ref());
        c.op = Some(Op::Cpy);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_permute(
        &mut self,
        a: Arc<Tensor<T>>,
        axis0: usize,
        axis1: usize,
        axis2: usize,
        axis3: usize,
    ) -> Tensor<T> {
        assert!(axis0 < MAX_DIMS);
        assert!(axis1 < MAX_DIMS);
        assert!(axis2 < MAX_DIMS);
        assert!(axis3 < MAX_DIMS);

        assert!(axis0 != axis1);
        assert!(axis0 != axis2);
        assert!(axis0 != axis3);
        assert!(axis1 != axis2);
        assert!(axis1 != axis3);
        assert!(axis2 != axis3);

        assert!(a.grad.is_none());

        let mut c = self.view_tensor(a.as_ref());
        c.ne[axis0] = a.ne[0];
        c.ne[axis1] = a.ne[1];
        c.ne[axis2] = a.ne[2];
        c.ne[axis3] = a.ne[3];

        c.nb[axis0] = a.nb[0];
        c.nb[axis1] = a.nb[1];
        c.nb[axis2] = a.nb[2];
        c.nb[axis3] = a.nb[3];

        c.op = Some(Op::Permute);
        c.src0 = Some(a.clone());
        c
    }

    pub fn op_reshape(&mut self, a: Arc<Tensor<T>>, b: Arc<Tensor<T>>) -> Tensor<T> {
        assert!(a.is_contiguous());
        assert!(b.is_contiguous());
        assert!(a.nelements() == b.nelements());

        assert!(a.grad.is_none());
        assert!(b.grad.is_none());

        let mut c = self.new_tensor_with_data(b.n_dims, &b.ne, a.data.clone());
        c.op = Some(Op::Reshape);
        c.src0 = Some(a.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_reshape_2d(&mut self, a: Arc<Tensor<T>>, ne0: usize, ne1: usize) -> Tensor<T> {
        assert!(a.is_contiguous());
        assert!(a.nelements() == ne0 * ne1);

        assert!(a.grad.is_none());

        let ne = [ne0, ne1];
        let mut c = self.new_tensor_with_data(ne.len(), &ne, a.data.clone());
        c.op = Some(Op::Reshape);
        c.src0 = Some(a.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    pub fn op_reshape_3d(
        &mut self,
        a: Arc<Tensor<T>>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Tensor<T> {
        assert!(a.is_contiguous());
        assert!(a.nelements() == ne0 * ne1 * ne2);

        assert!(a.grad.is_none());

        let ne = [ne0, ne1, ne2];
        let mut c = self.new_tensor_with_data(ne.len(), &ne, a.data.clone());
        c.op = Some(Op::Reshape);
        c.src0 = Some(a.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(self.dup_tensor(a.as_ref())));
        }
        c
    }

    // TODO: Check if mode type is correct
    pub fn op_rope(
        &mut self,
        a: Arc<Tensor<T>>,
        n_past: usize,
        n_dims: usize,
        mode: usize,
    ) -> Tensor<T> {
        assert!(a.grad.is_none());

        let b = Arc::new(self.new_tensor_1d(3));
        let mut data = b.data.lock().unwrap();
        data[0] = T::from(n_past);
        data[1] = T::from(n_dims);
        data[2] = T::from(mode);

        let mut c = self.view_tensor(a.as_ref());
        c.op = Some(Op::Rope);
        c.src0 = Some(a.clone());
        c.src1 = Some(b.clone());
        c
    }
}
