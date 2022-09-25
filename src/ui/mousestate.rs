use crate::ui::mousebuttonstate::MouseButtonState;

pub enum MouseState {
    Up(MouseButtonState),
    Down(MouseButtonState),
    Nothing,
}
