//! # ggml-rs
//! ggml-rs is a Rust implementation of the [ggml](https://github.com/ggerganov/ggml), a tensor library for machine learning used by llama.cpp and whisper.cpp.

use std::{
    cmp::min,
    marker::PhantomData,
    mem::size_of,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
};

use num_traits::Num;

const MAX_DIMS: usize = 4;
const MAX_NODES: usize = 4096;
const MAX_OPT: usize = 4;

#[derive(Debug, Clone)]
pub enum Op {
    Dup,
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
pub struct Tensor<T: Send + Sync + Num + Copy + From<usize>> {
    n_dims: usize,

    ne: [usize; MAX_DIMS],
    nb: [usize; MAX_DIMS],

    op: Option<Op>,

    #[allow(dead_code)]
    is_param: bool,

    grad: Option<Arc<Mutex<Tensor<T>>>>,
    src0: Option<Arc<Mutex<Tensor<T>>>>,
    src1: Option<Arc<Mutex<Tensor<T>>>>,
    opt: [Option<Arc<Mutex<Tensor<T>>>>; MAX_OPT],

    n_tasks: usize,

    data: Arc<Mutex<Vec<T>>>,
}

impl<T: Send + Sync + Num + Copy + From<usize>> Tensor<T> {
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
            n_dims,
            ne,
            nb: [0; MAX_DIMS],
            op: None,
            is_param: false,
            grad: None,
            src0: None,
            src1: None,
            opt: [None, None, None, None],
            n_tasks: 0,
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

pub struct Context<T: Send + Sync + Num + Copy + From<usize>> {
    r#type: PhantomData<T>,
}

impl<T: Send + Sync + Num + Copy + From<usize>> Context<T> {
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

    pub fn op_get_rows(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        _b: Arc<Mutex<Tensor<T>>>,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();
        assert!(a.is_matrix() && b.is_vector());

        let mut c = self.new_tensor_2d(a.ne[0], b.ne[0]);
        c.op = Some(Op::GetRows);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    fn op_unary_op(&mut self, op: Op, _a: Arc<Mutex<Tensor<T>>>, inplace: bool) -> Tensor<T> {
        assert!(op.is_unary_op());

        let a = _a.lock().unwrap();

        let mut c = if inplace {
            self.view_tensor(&a)
        } else {
            self.dup_tensor(&a)
        };
        c.op = Some(op);
        c.src0 = Some(_a.clone());
        if !inplace && a.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    fn op_binary_op(
        &mut self,
        op: Op,
        _a: Arc<Mutex<Tensor<T>>>,
        _b: Arc<Mutex<Tensor<T>>>,
        inplace: bool,
    ) -> Tensor<T> {
        assert!(op.is_binary_op());

        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();
        assert!(a.is_same_shape(&b));

        let mut c = if inplace {
            self.view_tensor(&a)
        } else {
            self.dup_tensor(&a)
        };
        c.op = Some(op);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        if !inplace && (a.grad.is_some() || b.grad.is_some()) {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_add(&mut self, a: Arc<Mutex<Tensor<T>>>, b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_binary_op(Op::Add, a, b, false)
    }

    pub fn op_sub(&mut self, a: Arc<Mutex<Tensor<T>>>, b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_binary_op(Op::Sub, a, b, false)
    }

    pub fn op_mul(&mut self, a: Arc<Mutex<Tensor<T>>>, b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_binary_op(Op::Mul, a, b, false)
    }

    pub fn op_div(&mut self, a: Arc<Mutex<Tensor<T>>>, b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_binary_op(Op::Div, a, b, false)
    }

    pub fn op_norm(&mut self, a: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_unary_op(Op::Norm, a, false)
    }

    pub fn op_rms_norm(&mut self, a: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_unary_op(Op::RmsNorm, a, false)
    }

    pub fn op_silu(&mut self, a: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        self.op_unary_op(Op::Silu, a, false)
    }

    pub fn op_repeat(&mut self, _a: Arc<Mutex<Tensor<T>>>, _b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();

        assert!(a.can_repeat(&b));

        let is_node = if a.grad.is_some() { true } else { false };

        if a.is_same_shape(&b) && !is_node {
            // TODO: return a.clone()
        }

        let mut c = self.new_tensor(b.n_dims, &b.ne);
        c.op = Some(Op::Repeat);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        if is_node {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_mul_mat(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        _b: Arc<Mutex<Tensor<T>>>,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();

        assert!(a.can_mul_mat(&b));
        assert!(a.is_transposed() == false);

        let n_dims = min(a.n_dims, b.n_dims);
        let ne = [a.ne[1], b.ne[1], a.ne[2], b.ne[3]];
        let mut c = self.new_tensor(n_dims, &ne);
        c.op = Some(Op::MulMat);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_scale(&mut self, _a: Arc<Mutex<Tensor<T>>>, _b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();

        assert!(b.is_scalar());
        assert!(a.is_padded_1d());

        let mut c = self.view_tensor(&a);
        c.op = Some(Op::Scale);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_diag_mask_inf(&mut self, _a: Arc<Mutex<Tensor<T>>>, n_past: usize) -> Tensor<T> {
        let a = _a.lock().unwrap();

        let b = Arc::new(Mutex::new(self.new_scalar(n_past.into())));

        let mut c = self.view_tensor(&a);
        c.op = Some(Op::DiagMaskInf);
        c.src0 = Some(_a.clone());
        c.src1 = Some(b.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_soft_max(&mut self, _a: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        let a = _a.lock().unwrap();

        let mut c = self.view_tensor(&a);
        c.op = Some(Op::SoftMax);
        c.src0 = Some(_a.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_view_1d(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        ne0: usize,
        offset: usize,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();

        assert!(a.grad.is_none());

        let mut c = self.new_tensor_1d(ne0);
        c.op = Some(Op::View);
        c.src0 = Some(_a.clone());
        c.src1 = Some(Arc::new(Mutex::new(self.new_scalar(offset.into()))));
        c
    }

    pub fn op_view_2d(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();

        assert!(a.grad.is_none());

        let mut c = self.new_tensor_2d(ne0, ne1);
        c.nb[1] = nb1;
        c.nb[2] = c.nb[1] * ne1;
        c.nb[3] = c.nb[2];

        c.op = Some(Op::View);
        c.src0 = Some(_a.clone());
        c.src1 = Some(Arc::new(Mutex::new(self.new_scalar(offset.into()))));
        c
    }

    pub fn op_view_3d(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();

        assert!(a.grad.is_none());

        let mut c = self.new_tensor_3d(ne0, ne1, ne2);
        c.nb[1] = nb1;
        c.nb[2] = nb2;
        c.nb[3] = c.nb[2] * ne2;

        c.op = Some(Op::View);
        c.src0 = Some(_a.clone());
        c.src1 = Some(Arc::new(Mutex::new(self.new_scalar(offset.into()))));
        c
    }

    pub fn op_cpy(&mut self, _a: Arc<Mutex<Tensor<T>>>, _b: Arc<Mutex<Tensor<T>>>) -> Tensor<T> {
        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();

        assert!(a.nelements() == b.nelements());

        let mut c = self.view_tensor(&b);
        c.op = Some(Op::Cpy);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_permute(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
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

        let a = _a.lock().unwrap();

        assert!(a.grad.is_none());

        let mut c = self.view_tensor(&a);
        c.ne[axis0] = a.ne[0];
        c.ne[axis1] = a.ne[1];
        c.ne[axis2] = a.ne[2];
        c.ne[axis3] = a.ne[3];

        c.nb[axis0] = a.nb[0];
        c.nb[axis1] = a.nb[1];
        c.nb[axis2] = a.nb[2];
        c.nb[axis3] = a.nb[3];

        c.op = Some(Op::Permute);
        c.src0 = Some(_a.clone());
        c
    }

    pub fn op_reshape(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        _b: Arc<Mutex<Tensor<T>>>,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();
        let b = _b.lock().unwrap();

        assert!(a.is_contiguous());
        assert!(b.is_contiguous());
        assert!(a.nelements() == b.nelements());

        assert!(a.grad.is_none());
        assert!(b.grad.is_none());

        let mut c = self.new_tensor_with_data(b.n_dims, &b.ne, a.data.clone());
        c.op = Some(Op::Reshape);
        c.src0 = Some(_a.clone());
        if a.grad.is_some() || b.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_reshape_2d(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        ne0: usize,
        ne1: usize,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();

        assert!(a.is_contiguous());
        assert!(a.nelements() == ne0 * ne1);

        assert!(a.grad.is_none());

        let ne = [ne0, ne1];
        let mut c = self.new_tensor_with_data(ne.len(), &ne, a.data.clone());
        c.op = Some(Op::Reshape);
        c.src0 = Some(_a.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    pub fn op_reshape_3d(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();

        assert!(a.is_contiguous());
        assert!(a.nelements() == ne0 * ne1 * ne2);

        assert!(a.grad.is_none());

        let ne = [ne0, ne1, ne2];
        let mut c = self.new_tensor_with_data(ne.len(), &ne, a.data.clone());
        c.op = Some(Op::Reshape);
        c.src0 = Some(_a.clone());
        if a.grad.is_some() {
            c.grad = Some(Arc::new(Mutex::new(self.dup_tensor(&a))));
        }
        c
    }

    // TODO: Check if mode type is correct
    pub fn op_rope(
        &mut self,
        _a: Arc<Mutex<Tensor<T>>>,
        n_past: usize,
        n_dims: usize,
        mode: usize,
    ) -> Tensor<T> {
        let a = _a.lock().unwrap();

        assert!(a.grad.is_none());

        let _b = Arc::new(Mutex::new(self.new_tensor_1d(3)));
        let b = _b.lock().unwrap();
        let mut data = b.data.lock().unwrap();
        data[0] = T::from(n_past);
        data[1] = T::from(n_dims);
        data[2] = T::from(mode);

        let mut c = self.view_tensor(&a);
        c.op = Some(Op::Rope);
        c.src0 = Some(_a.clone());
        c.src1 = Some(_b.clone());
        c
    }

    pub fn graph_compute(&mut self, graph: &mut ComputationGraph<T>) {
        let n_threads = graph.n_threads;

        let state_shared = Arc::new(ComputeStateShared {
            n_threads: n_threads,
            n_ready: AtomicUsize::new(0),
            has_work: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });

        thread::scope(|s| {
            let mut workers = Vec::with_capacity(n_threads - 1);

            for i in 0..graph.n_threads - 1 {
                let state = ComputeState::<T>::new(i + 1, n_threads, state_shared.clone());
                let worker = s.spawn(move || loop {
                    if state.shared.n_ready.fetch_add(1, Ordering::SeqCst)
                        == state.shared.n_threads - 1
                    {
                        state.shared.has_work.store(false, Ordering::SeqCst);
                    } else {
                        while !state.shared.has_work.load(Ordering::SeqCst) {
                            if state.shared.stop.load(Ordering::SeqCst) {
                                return;
                            }
                        }
                    }

                    state.shared.n_ready.fetch_sub(1, Ordering::SeqCst);

                    while !state.shared.has_work.load(Ordering::SeqCst) {
                        if state.shared.stop.load(Ordering::SeqCst) {
                            return;
                        }
                    }

                    if state.shared.stop.load(Ordering::SeqCst) {
                        return;
                    }

                    if let Some(node) = state.node.as_ref() {
                        state.params.compute_forward(node.as_ref());
                    }
                });
                workers.push(worker);
            }

            // initialize tasks + work buffer
            // thread scheduling for the different operations
            for node in graph.nodes.iter() {
                let mut node = node.lock().unwrap();
                if let Some(op) = &node.op {
                    match op {
                        Op::Dup => {
                            node.n_tasks = 1;
                        }
                        Op::Add => {
                            node.n_tasks = n_threads;
                        }
                        Op::Sub
                        | Op::Mul
                        | Op::Div
                        | Op::Sqr
                        | Op::Sqrt
                        | Op::Sum
                        | Op::Mean
                        | Op::Repeat
                        | Op::Abs
                        | Op::Sgn
                        | Op::Neg
                        | Op::Step
                        | Op::Relu => {
                            node.n_tasks = 1;
                        }
                        Op::Gelu => {
                            node.n_tasks = n_threads;
                        }
                        Op::Silu => {
                            node.n_tasks = n_threads;
                        }
                        Op::Norm | Op::RmsNorm => {
                            node.n_tasks = n_threads;
                        }
                        Op::MulMat => {
                            node.n_tasks = n_threads;
                            // TODO: Implement
                        }
                        Op::Scale => {
                            node.n_tasks = n_threads;
                        }
                        Op::Cpy
                        | Op::Reshape
                        | Op::View
                        | Op::Permute
                        | Op::Transpose
                        | Op::GetRows
                        | Op::DiagMaskInf => {
                            node.n_tasks = 1;
                        }
                        Op::SoftMax => {
                            node.n_tasks = n_threads;
                        }
                        Op::Rope => {
                            node.n_tasks = n_threads;
                        }
                        Op::Conv1D1S | Op::Conv1D2S => {
                            // TODO: Implement
                        }
                        Op::FlashAttn => {
                            // TODO: Implement
                        }
                        Op::FlashFF => {
                            // TODO: Implement
                        }
                    }
                } else {
                    node.n_tasks = 1;
                }

                // TODO: Setup work buffer
            }

            for node in graph.nodes.iter() {
                let node = node.lock().unwrap();

                // INIT
                let mut params = ComputeParams {
                    task_type: TaskType::Init,
                    ith: 0,
                    nth: node.n_tasks,
                };

                params.compute_forward(&node);

                // COMPUTE
                if node.n_tasks > 1 {
                    if state_shared.n_ready.fetch_add(1, Ordering::SeqCst)
                        == state_shared.n_threads - 1
                    {
                        state_shared.has_work.store(false, Ordering::SeqCst);
                    }
                    while state_shared.has_work.load(Ordering::SeqCst) {
                        // Do Nothing
                    }

                    // launch thread pool
                    for i in 0..n_threads - 1 {
                        let params = ComputeParams {
                            task_type: TaskType::Compute,
                            ith: i + 1,
                            nth: node.n_tasks,
                        };
                        // TODO
                    }

                    state_shared.n_ready.fetch_sub(1, Ordering::SeqCst);

                    while state_shared.n_ready.load(Ordering::SeqCst) > 0 {
                        // Do Nothing
                    }

                    state_shared.has_work.store(true, Ordering::SeqCst);
                }

                params.task_type = TaskType::Finalize;
                params.compute_forward(&node);

                // wait for thread pool
                if node.n_tasks > 1 {
                    if state_shared.n_ready.fetch_add(1, Ordering::SeqCst) == n_threads - 1 {
                        state_shared.has_work.store(false, Ordering::SeqCst);
                    }

                    while state_shared.has_work.load(Ordering::SeqCst) {
                        // Do Nothing
                    }

                    state_shared.n_ready.fetch_sub(1, Ordering::SeqCst);

                    while state_shared.n_ready.load(Ordering::SeqCst) > 0 {
                        // Do Nothing
                    }
                }

                // FINALIZE
                if node.n_tasks > 1 {
                    if state_shared.n_ready.fetch_add(1, Ordering::SeqCst) == n_threads - 1 {
                        state_shared.has_work.store(false, Ordering::SeqCst);
                    }

                    while state_shared.has_work.load(Ordering::SeqCst) {
                        // Do Nothing
                    }

                    // launch thread pool
                    for i in 0..n_threads - 1 {
                        let params = ComputeParams {
                            task_type: TaskType::Finalize,
                            ith: i + 1,
                            nth: node.n_tasks,
                        };
                        // TODO
                    }

                    state_shared.n_ready.fetch_sub(1, Ordering::SeqCst);

                    while state_shared.n_ready.load(Ordering::SeqCst) > 0 {
                        // Do Nothing
                    }

                    state_shared.has_work.store(true, Ordering::SeqCst);
                }

                params.task_type = TaskType::Finalize;
                params.compute_forward(&node);

                // wait for thread pool
                if node.n_tasks > 1 {
                    if state_shared.n_ready.fetch_add(1, Ordering::SeqCst) == n_threads - 1 {
                        state_shared.has_work.store(false, Ordering::SeqCst);
                    }

                    while state_shared.has_work.load(Ordering::SeqCst) {
                        // Do Nothing
                    }

                    state_shared.n_ready.fetch_sub(1, Ordering::SeqCst);

                    while state_shared.n_ready.load(Ordering::SeqCst) > 0 {
                        // Do Nothing
                    }
                }

                // TODO: performance states (node)
            }

            // join thread pool
            if n_threads > 1 {
                state_shared.stop.store(true, Ordering::SeqCst);
                state_shared.has_work.store(true, Ordering::SeqCst);

                for worker in workers {
                    worker.join().unwrap();
                }
            }

            // TODO: performance states (graph)
        })
    }
}

enum TaskType {
    Init,
    Compute,
    Finalize,
}

struct ComputeParams {
    task_type: TaskType,
    ith: usize,
    nth: usize,
    // TODO: Add wsize
    // TODO: Add wdata
}

impl ComputeParams {
    fn compute_forward<T: Send + Sync + Num + Copy + From<usize>>(&self, node: &Tensor<T>) {}
}

struct ComputeStateShared {
    n_threads: usize,
    n_ready: AtomicUsize,
    has_work: AtomicBool,
    stop: AtomicBool,
}

struct ComputeState<T: Send + Sync + Num + Copy + From<usize>> {
    params: ComputeParams,
    node: Option<Arc<Tensor<T>>>,
    shared: Arc<ComputeStateShared>,
}

impl<T: Send + Sync + Num + Copy + From<usize>> ComputeState<T> {
    fn new(ith: usize, nth: usize, shared: Arc<ComputeStateShared>) -> Self {
        Self {
            params: ComputeParams {
                task_type: TaskType::Compute,
                ith,
                nth,
            },
            node: None,
            shared: shared,
        }
    }
}

pub struct ComputationGraph<T: Send + Sync + Num + Copy + From<usize>> {
    n_threads: usize,

    work: Vec<Arc<Mutex<Tensor<T>>>>,

    nodes: Vec<Arc<Mutex<Tensor<T>>>>,
    grads: Vec<Arc<Mutex<Tensor<T>>>>,
    leafs: Vec<Arc<Mutex<Tensor<T>>>>,
}

impl<T: Send + Sync + Num + Copy + From<usize>> ComputationGraph<T> {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads,
            work: Vec::new(),
            nodes: Vec::new(),
            grads: Vec::new(),
            leafs: Vec::new(),
        }
    }

    pub fn build_forward_expand(&mut self, tensor: Arc<Mutex<Tensor<T>>>) {
        let n0 = self.nodes.len();

        self.visit_parents(tensor.clone());

        let n1 = self.nodes.len();
        eprintln!("visited {} new nodes", n1 - n0);

        if n1 - n0 > 0 {
            assert!(Arc::ptr_eq(self.nodes.last().unwrap(), &tensor));
        }
    }

    fn visit_parents(&mut self, node: Arc<Mutex<Tensor<T>>>) {
        let locked_node = node.lock().unwrap();
        if locked_node.grad.is_none() {
            if let Some(op) = &locked_node.op {
                eprintln!("node {:p} has no grad, but op {:?}", node, op);
            }
        }

        for n in self.nodes.iter() {
            if Arc::ptr_eq(n, &node) {
                return;
            }
        }

        for n in self.leafs.iter() {
            if Arc::ptr_eq(n, &node) {
                return;
            }
        }

        if locked_node.src0.is_some() {
            self.visit_parents(locked_node.src0.as_ref().unwrap().clone());
            self.visit_parents(locked_node.src0.as_ref().unwrap().clone());
        }

        if locked_node.src1.is_some() {
            self.visit_parents(locked_node.src1.as_ref().unwrap().clone());
        }

        for opt in locked_node.opt.iter() {
            if let Some(opt) = opt {
                self.visit_parents(opt.clone());
            }
        }

        if locked_node.op.is_none() && locked_node.grad.is_none() {
            assert!(self.leafs.len() < MAX_NODES);

            self.leafs.push(node.clone());
        } else {
            assert!(self.nodes.len() < MAX_NODES);

            self.nodes.push(node.clone());
            self.grads.push(locked_node.grad.as_ref().unwrap().clone());
        }
    }
}
