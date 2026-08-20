# Como falar com o gemma4:26b no teu PC

O Ollama corre como um servidor local em `http://localhost:11434`. O modelo
só ocupa RAM quando é usado e descarrega-se sozinho passados ~5 min de
inatividade.

## Via terminal (o mais simples)

```powershell
# conversa interativa (escreve /bye para sair)
ollama run gemma4:26b

# pergunta única, resposta direta no terminal
ollama run gemma4:26b "Resume isto em 3 pontos: ..."

# passar um ficheiro inteiro como contexto
Get-Content reuniao.txt | ollama run gemma4:26b "Resume esta reunião"

# gestão
ollama list          # modelos instalados
ollama ps            # o que está carregado em RAM agora
ollama stop gemma4:26b   # descarregar da RAM já
```

Se `ollama` não for reconhecido, abre um terminal novo (o PATH é adicionado
na instalação) ou usa o caminho completo:
`C:\Users\Utilizador\AppData\Local\Programs\Ollama\ollama.exe`

## Via API HTTP (para scripts e integrações)

Endpoint principal: `POST http://localhost:11434/api/generate`

```powershell
# PowerShell
$body = @{ model = "gemma4:26b"; prompt = "Explica MoE em 2 frases"; stream = $false } | ConvertTo-Json
Invoke-RestMethod http://localhost:11434/api/generate -Method Post -Body $body -ContentType "application/json" | Select-Object -ExpandProperty response
```

```bash
# curl
curl http://localhost:11434/api/generate -d '{
  "model": "gemma4:26b",
  "prompt": "Explica MoE em 2 frases",
  "stream": false
}'
```

Para conversas com histórico usa `/api/chat`:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4:26b",
  "messages": [
    {"role": "system", "content": "Responde sempre em PT-PT."},
    {"role": "user", "content": "O que é WASAPI loopback?"}
  ],
  "stream": false
}'
```

```python
# Python (pip install ollama)
import ollama
resposta = ollama.chat(model="gemma4:26b", messages=[
    {"role": "user", "content": "Resume esta reunião: ..."}
])
print(resposta["message"]["content"])
```

A API é compatível com o formato OpenAI em `http://localhost:11434/v1`
(`/v1/chat/completions`), portanto qualquer ferramenta que aceite uma
"OpenAI base URL" funciona com `api_key` qualquer e `model: gemma4:26b`.

## Notas para esta máquina

- Sem GPU: o modelo corre em CPU. Como é MoE com só 4B parâmetros ativos,
  a velocidade é a de um modelo pequeno (~5-15 tokens/s), com qualidade de
  um 26B.
- Ocupa ~18-20 GB de RAM quando carregado (tens 32 GB — evita tê-lo
  carregado durante a transcrição de uma reunião longa; o meetrec só o
  chama depois da transcrição terminar).
- O primeiro pedido depois de idle demora ~30-60 s (carregar 18 GB do disco
  para a RAM); os seguintes são imediatos.
- O meetrec usa exatamente esta API (`summary.backend: ollama` no
  config.yaml) como fallback do Gemini.
