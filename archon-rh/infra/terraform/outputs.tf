output "namespace" {
  value = kubernetes_namespace.archon.metadata[0].name
}
