import Lake
open Lake DSL

package «archon-rh» where
  moreLeanArgs := #["-Dlinter.all=true"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
  @ "v4.10.0"

@[defaultTarget]
lean_lib ArchonRH where
  roots := #[`ArchonRiemann, `ArchonCertificates]
