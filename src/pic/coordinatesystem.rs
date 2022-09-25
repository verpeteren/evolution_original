use std::fmt::{Display, Formatter, Result as FResult};
use std::ops::Not;
use std::str::FromStr;

use clap::ArgEnum;
use variant_count::VariantCount;

#[derive(Clone, Debug, PartialEq, ArgEnum, VariantCount)]
pub enum CoordinateSystem {
    Polar,
    Cartesian,
}

impl CoordinateSystem {
    pub fn list_all<'a>() -> Vec<String> {
        vec![CoordinateSystem::Polar.to_string(), CoordinateSystem::Cartesian.to_string()]
    }
}

impl Display for CoordinateSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> FResult {
        let x = match self {
            CoordinateSystem::Polar => "polar",
            CoordinateSystem::Cartesian => "cartesian",
        };
        write!(f, "{}", x)
    }
}

pub const DEFAULT_COORDINATE_SYSTEM: CoordinateSystem = CoordinateSystem::Polar;

impl FromStr for CoordinateSystem {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_ref() {
            "polar" => Ok(CoordinateSystem::Polar),
            "cartesian" => Ok(CoordinateSystem::Cartesian),
            _ => Err(format!("Cannot parse {}. Not a known coordinate system", s)),
        }
    }
}

impl Not for CoordinateSystem {
    type Output = Self;
    fn not(self) -> Self::Output {
        match self {
            CoordinateSystem::Polar => CoordinateSystem::Cartesian,
            CoordinateSystem::Cartesian => CoordinateSystem::Polar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordsystem_parse() {
        assert_eq!("Polar".parse(), Ok(CoordinateSystem::Polar));
        assert_eq!("PoLar".parse(), Ok(CoordinateSystem::Polar));
        assert_eq!("POLAR".parse(), Ok(CoordinateSystem::Polar));
        assert_eq!("cartesian".parse(), Ok(CoordinateSystem::Cartesian));
        assert_eq!("Cartesian".parse(), Ok(CoordinateSystem::Cartesian));
        assert_eq!("CARTESIAN".parse(), Ok(CoordinateSystem::Cartesian));
        assert_eq!(
            "mercator".parse::<CoordinateSystem>(),
            Err("Cannot parse mercator. Not a known coordinate system".to_string())
        );
    }

    #[test]
    fn test_coordsystem_not() {
        assert_eq!(!CoordinateSystem::Polar, CoordinateSystem::Cartesian);
        assert_eq!(!CoordinateSystem::Cartesian, CoordinateSystem::Polar);
    }

    #[test]
    fn test_coordsystem_display() {
        assert_eq!(&CoordinateSystem::Polar.to_string(), "polar");
        assert_eq!(&CoordinateSystem::Cartesian.to_string(), "cartesian");
    }
}