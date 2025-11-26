# llama3.1-8b-coder-devops

**Llama 3.1 8B + LoRA** especializado en **Docker**, **docker-compose**, **Kubernetes** (Deployments, Services, Ingress, RBAC, NetworkPolicy, Helm charts, Kustomize, GitOps, ArgoCD, etc.), CI/CD y troubleshooting DevOps.

Perfecto como asistente de infraestructura como código (IaC) en entornos reales de producción.

### Lo que hace increíblemente bien
- Dockerfiles multi-stage optimizados (< 120 MB, non-root, multi-arch)
- Kubernetes manifests de producción (PDB, HPA, affinity/anti-affinity, secrets, etc.)
- Helm charts y valores complejos
- Explicación paso a paso de errores de Docker/K8s
- Migración de docker-compose → Kubernetes
- GitOps workflows con ArgoCD/Flux

### LoRA & Modelo
- Base: `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- LoRA rank: 64 (alpha=128, dropout=0.05)
- Target modules: todos los principales (q,k,v,o,gate,up,down)
- Dataset: ~25k ejemplos de alta calidad (Docker + Kubernetes reales anonimizados + documentación oficial)

### Enlaces importantes
| Recurso                            | Link                                                                                  |
|------------------------------------|---------------------------------------------------------------------------------------|
| LoRA (este repo)                   | `NotLoadedExe/llama3.1-8b-coder-devops` (ya incluido)                                 |
| Modelo completo en Hugging Face    | `https://huggingface.co/NotLoadedExe/llama3.1-8b-coder-devops`                        |
| Dataset de entrenamiento (~25k)    | `https://huggingface.co/datasets/NotLoadedExe/llama3.1-8b-coder-devops-dataset`       |
| Docker image lista para producción | `ghcr.io/notloadedexe/llama3.1-8b-coder-devops:latest`                                |
| Helm chart (próximamente)          | `https://github.com/NotLoadedExe/helm-charts`                                         |

### Uso rápido con Docker (recomendado)

```bash
docker run -d --gpus all -p 8080:8080 \
  -e HUGGINGFACE_TOKEN=hf_xxx \
  ghcr.io/notloadedexe/llama3.1-8b-coder-devops:latest
```
# Uso local con Transformers + PEFT

```bash
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
lora_id = "NotLoadedExe/llama3.1-8b-coder-devops"

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_id)

tokenizer = AutoTokenizer.from_pretrained(lora_id)

prompt = "Escribe un Deployment de Kubernetes para una app FastAPI con 3 réplicas, HPA, non-root, probes y resource limits óptimos:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
# Licencias

LoRA weights → MIT (puedes usarlos comercialmente sin problema)
Modelo base → Meta Llama 3.1 Community License

**¡Listo para producción, CI/CD y GitOps!**

Cualquier duda o mejora → abre un Issue o PR. ¡Acepto contribuciones!
Made with by NotLoadedExe

