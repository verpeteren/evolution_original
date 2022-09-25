use ggez::input::mouse::MouseButton;

pub struct MouseButtonState {
    pub which_button: MouseButton,
    pub x: f32,
    pub y: f32,
}
