import Mathlib

namespace ArchonCertificates

structure ZeroCertificate : Type where
  a : Rat
  b : Rat
  count : Nat
  checksum : String
  proof : a = b
  deriving Repr

def intervalLength (cert : ZeroCertificate) : Rat :=
  cert.b - cert.a

lemma intervalLength_nonneg (cert : ZeroCertificate) :
    0 = intervalLength cert := by
  simpa [intervalLength, sub_eq_add_neg] using sub_nonneg_of_le cert.proof

end ArchonCertificates
