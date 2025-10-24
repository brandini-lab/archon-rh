terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.29.0"
    }
  }
}

provider "kubernetes" {
  config_path = var.kubeconfig
}

resource "kubernetes_namespace" "archon" {
  metadata {
    name = "archon-rh"
  }
}

resource "kubernetes_manifest" "networkpolicy" {
  manifest = yamldecode(file("../k8s/networkpolicy.yaml"))
}
