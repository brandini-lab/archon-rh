import Lake
open Lake DSL

package ArchonLean where
  moreLeanArgs := #["-Dlinter.all=true"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
  @ "v4.10.0"

@[defaultTarget]
lean_lib ArchonLean where
  roots := #[`Mathlib]
