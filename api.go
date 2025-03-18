package nngo

// Add is the user-facing API function for adding tensors
func Add[T Number](self, other *Tensor[T]) *Tensor[T] {
    ctx := &Context[T]{}
    op := &AddFn[T]{ Ctx: ctx }
    result, err := op.Forward(ctx, self, other)
    if err != nil { panic("error in Add operation") }
    if self.RequiresGrad || other.RequiresGrad {
        result.RequiresGrad = true
        result.GradFn = op
    }
    return result
}

// Method version of Add attached to Tensor struct
func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] { return Add(t, other) }
