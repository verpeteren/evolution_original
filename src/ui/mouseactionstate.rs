#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub struct MouseActionState {
    pub pos: (i32, i32),
    pub pressed: (bool, bool, bool),
    pub wheel: f32,
}
