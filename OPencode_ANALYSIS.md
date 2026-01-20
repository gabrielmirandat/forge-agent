# Análise do OpenCode vs Forge Agent

## 📋 Visão Geral

### OpenCode
- **Tipo**: Agente de código AI open-source
- **Interface**: Terminal UI (TUI) + Desktop App + Web Console
- **Arquitetura**: Cliente/Servidor (server local + múltiplos clientes)
- **Linguagem**: TypeScript/Bun
- **Foco**: Terminal-first, experiência de desenvolvedor

### Forge Agent
- **Tipo**: Agente de código AI
- **Interface**: Web UI (React/Vite)
- **Arquitetura**: API REST (FastAPI) + Frontend
- **Linguagem**: Python (backend) + TypeScript (frontend)
- **Foco**: Acesso via web/mobile, multi-sessão

---

## 🏗️ Arquitetura

### OpenCode

```
┌─────────────────────────────────────────────────────────┐
│                    OpenCode Server                       │
│  (Hono/Bun - roda localmente na máquina do dev)         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Sessions   │  │   Projects   │  │    Tools     │  │
│  │  (Storage)   │  │  (Git/VCS)   │  │  (Registry)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │     LSP      │  │     PTY      │  │   Snapshot  │  │
│  │  (Language   │  │  (Terminal   │  │  (Git diff) │  │
│  │   Server)    │  │   Sessions)  │  │             │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   TUI Client │  │ Desktop App  │  │ Web Console │
│  (Terminal)  │  │   (Tauri)    │  │  (SolidJS)  │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Características principais:**
- **Server local**: Roda na máquina do desenvolvedor
- **Múltiplos clientes**: TUI, Desktop, Web podem se conectar ao mesmo server
- **PTY (Pseudo-Terminal)**: Cada sessão tem um PTY persistente para comandos shell
- **Storage baseado em arquivos**: JSON files em `~/.opencode/storage/`
- **Projeto = Git repo**: Identifica projetos pelo commit root do Git

### Forge Agent

```
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python)                   │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Sessions   │  │   Planner   │  │  Executor   │  │
│  │  (SQLite)    │  │   (LLM)     │  │  (Tools)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Tools     │  │   Storage    │  │   Tmux      │  │
│  │  (Registry)   │  │  (SQLite)    │  │  (Sessions) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              React Frontend (Vite)                       │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  ChatPage    │  │  Components  │  │   API Client │  │
│  │  (Sessions)  │  │  (Viewers)   │  │   (HTTP)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Características principais:**
- **API REST**: Backend separado do frontend
- **SQLite**: Banco de dados para sessões e mensagens
- **Tmux**: Sessões persistentes para contexto de shell
- **Web-first**: Acesso via navegador/mobile

---

## 🔑 Diferenças Principais

### 1. **Execução de Comandos**

#### OpenCode
- **PTY (Pseudo-Terminal)**: Usa `bun-pty` para criar terminais virtuais
- **Sessões PTY persistentes**: Cada sessão pode ter múltiplos PTYs
- **WebSocket**: Clientes se conectam via WebSocket para interagir com PTY
- **Comandos executam diretamente**: `spawn()` com shell nativo
- **Contexto mantido**: Cada PTY mantém seu próprio estado (cwd, env vars)

```typescript
// opencode/packages/opencode/src/pty/index.ts
export async function create(input: CreateInput) {
  const ptyProcess = spawn(command, args, {
    name: "xterm-256color",
    cwd,
    env,
  })
  // PTY persiste e mantém estado
}
```

#### Forge Agent
- **Tmux**: Usa sessões tmux para manter contexto
- **send_keys**: Executa comandos via `tmux send-keys`
- **Captura output**: Usa `capture-pane` para pegar output
- **Uma sessão tmux por sessão do agente**: Mapeamento 1:1

```python
# agent/tools/tmux.py
async def execute_command(self, session_name: str, command: str):
    pane.send_keys(command, enter=True)
    # Captura output do pane
    result = pane.cmd('capture-pane', '-p')
```

**Vantagem OpenCode**: PTY é mais leve e nativo, não precisa de tmux instalado
**Vantagem Forge Agent**: Tmux é mais comum em servidores Linux, permite reattach manual

---

### 2. **Gerenciamento de Memória/Contexto**

#### OpenCode
- **Sem vector DB**: Não usa embeddings ou vector database
- **Compaction**: Usa LLM para compactar conversas antigas em resumos
- **Summary**: Gera resumos de sessões e mensagens
- **Pruning**: Remove outputs de tool calls antigos quando necessário
- **Storage em arquivos**: JSON files organizados por projeto/sessão

```typescript
// opencode/packages/opencode/src/session/compaction.ts
export async function process(input: {
  messages: MessageV2.WithParts[]
  sessionID: string
}) {
  // Usa LLM para criar um resumo da conversa
  // Remove mensagens antigas, mantém apenas o resumo
}
```

**Estratégia de Compaction:**
1. Quando tokens excedem limite do modelo
2. Cria uma mensagem de "compaction" usando LLM
3. Resumo contém: o que foi feito, arquivos trabalhados, próximos passos
4. Remove mensagens antigas, mantém apenas o resumo

#### Forge Agent
- **Sem vector DB**: Também não usa (ainda)
- **SQLite**: Armazena todas as mensagens
- **Sem compaction**: Mantém todas as mensagens
- **Context limitado**: Usa últimas N mensagens no prompt

**Problema atual**: Pode exceder limites de contexto do modelo com sessões longas

---

### 3. **Estrutura de Dados**

#### OpenCode

```
Storage Structure:
storage/
  ├── session/
  │   └── {projectID}/
  │       └── {sessionID}.json    # Session metadata
  ├── message/
  │   └── {sessionID}/
  │       └── {messageID}.json    # Message metadata
  ├── part/
  │   └── {messageID}/
  │       └── {partID}.json       # Message parts (text, tool calls, etc.)
  └── session_diff/
      └── {sessionID}.json        # Git diffs da sessão
```

**Hierarquia:**
- **Project** → identificado pelo commit root do Git
- **Session** → pertence a um projeto, pode ter parentID (child sessions)
- **Message** → pertence a uma sessão, tem role (user/assistant)
- **Part** → partes de uma mensagem (text, tool-invocation, file, etc.)

#### Forge Agent

```
Database Structure:
sessions/
  ├── session_id (PK)
  ├── title
  ├── created_at
  ├── updated_at
  └── tmux_session        # Nome da sessão tmux

messages/
  ├── message_id (PK)
  ├── session_id (FK)
  ├── role
  ├── content
  ├── plan_result (JSON)
  └── execution_result (JSON)
```

**Hierarquia:**
- **Session** → sessão de chat
- **Message** → mensagem na sessão
- **Plan/Execution** → armazenados como JSON na mensagem

---

### 4. **Tools e Execução**

#### OpenCode

**Tools disponíveis:**
- `bash` - Executa comandos shell (com parsing de tree-sitter)
- `read` - Lê arquivos
- `write` - Escreve arquivos
- `edit` - Edita arquivos (multiedit)
- `grep` - Busca em arquivos
- `ls` - Lista diretórios
- `patch` - Aplica patches
- `lsp` - Integração com Language Server Protocol
- `codesearch` - Busca semântica (usando LSP)
- `skill` - Skills customizadas
- `task` - Gerenciamento de tarefas
- `websearch` - Busca web
- `webfetch` - Fetch de URLs

**Características:**
- **Tree-sitter parsing**: Parse de comandos bash para validação
- **Permission system**: Sistema de permissões granular
- **Tool registry**: Registry centralizado de tools
- **Tool metadata**: Tools podem retornar metadata além de output

#### Forge Agent

**Tools disponíveis:**
- `shell` - Executa comandos shell
- `filesystem` - Operações de arquivo (read, write, list, create, delete)
- `git` - Operações Git
- `github` - API GitHub + GitHub CLI
- `system` - Informações do sistema
- `tmux` - Gerenciamento de sessões tmux

**Características:**
- **Path validation**: Validação obrigatória de paths
- **Approval system**: Sistema de aprovação para operações destrutivas
- **Tmux integration**: Tudo executa no tmux quando session_id presente

---

## 🎯 Como OpenCode Resolve Problemas que Tivemos

### 1. **Persistência de Diretório (cd não funciona)**

**Problema**: Comandos `cd` não persistem entre execuções

**OpenCode:**
- Usa `workdir` parameter no tool bash
- Não executa `cd` - sempre passa `cwd` para `spawn()`
- Cada comando pode especificar seu próprio `workdir`

```typescript
// opencode/packages/opencode/src/tool/bash.ts
const proc = spawn(params.command, {
  shell,
  cwd: params.workdir || Instance.directory,  // Sempre especifica cwd
  // ...
})
```

**Forge Agent (nossa solução):**
- Usa tmux para manter contexto
- `cd` executa no tmux via `send_keys`
- Próximos comandos herdam o diretório do tmux

**Comparação:**
- **OpenCode**: Mais explícito, cada comando especifica onde executar
- **Forge Agent**: Mais implícito, contexto persiste automaticamente

---

### 2. **Gerenciamento de Contexto/Memória**

**Problema**: Sessões longas excedem limites de contexto do modelo

**OpenCode:**
- **Compaction automática**: Quando tokens excedem limite, compacta automaticamente
- **Summary**: Gera resumos de sessões e mensagens
- **Pruning**: Remove outputs de tool calls antigos

```typescript
// opencode/packages/opencode/src/session/compaction.ts
export async function isOverflow(input: {
  tokens: MessageV2.Assistant["tokens"]
  model: Provider.Model
}) {
  const context = input.model.limit.context
  const count = input.tokens.input + input.tokens.cache.read + input.tokens.output
  const usable = input.model.limit.input || context - output
  return count > usable  // Detecta overflow
}
```

**Forge Agent:**
- **Sem compaction**: Mantém todas as mensagens
- **Context limitado**: Usa apenas últimas N mensagens
- **Problema**: Pode perder contexto importante em sessões longas

**Solução recomendada para Forge Agent:**
- Implementar compaction similar ao OpenCode
- Usar LLM para gerar resumos de conversas antigas
- Manter apenas resumos + mensagens recentes

---

### 3. **Execução de Comandos**

**Problema**: Comandos precisam manter contexto (cwd, env vars)

**OpenCode:**
- **PTY persistente**: Cada sessão pode ter múltiplos PTYs
- **WebSocket streaming**: Output é streamed em tempo real
- **Buffer management**: Mantém buffer de output (2MB limit)

**Forge Agent:**
- **Tmux session**: Uma sessão tmux por sessão do agente
- **send_keys**: Executa comandos via tmux
- **capture-pane**: Captura output após execução

**Comparação:**
- **OpenCode PTY**: Mais leve, nativo, melhor para streaming
- **Forge Agent Tmux**: Mais comum em servidores, permite reattach manual

---

### 4. **Estrutura de Mensagens**

**OpenCode:**
- **MessageV2**: Estrutura rica com parts
- **Parts**: text, tool-invocation, file, reasoning, snapshot, patch
- **Hierarquia**: Message → Parts (múltiplos tipos)
- **Metadata**: Cada part pode ter metadata rica

```typescript
// opencode/packages/opencode/src/session/message-v2.ts
export const MessageV2 = {
  TextPart: { type: "text", text: string },
  ToolInvocationPart: { type: "tool-invocation", toolInvocation: {...} },
  FilePart: { type: "file", url: string, source: {...} },
  ReasoningPart: { type: "reasoning", text: string },
  SnapshotPart: { type: "snapshot", snapshot: string },
  PatchPart: { type: "patch", files: string[], hash: string },
}
```

**Forge Agent:**
- **Message simples**: role + content
- **Plan/Execution**: Armazenados como JSON na mensagem
- **Estrutura mais simples**: Menos flexível, mas mais direta

---

## 💡 Lições Aprendidas

### 1. **Compaction é Essencial**
OpenCode mostra que compaction automática é crucial para sessões longas. Sem isso, o contexto explode.

**Recomendação para Forge Agent:**
- Implementar compaction quando tokens excedem limite
- Usar LLM para gerar resumos
- Manter apenas resumos + mensagens recentes

### 2. **PTY vs Tmux**
Ambos funcionam, mas têm trade-offs:
- **PTY**: Mais leve, melhor para streaming, não precisa de dependência externa
- **Tmux**: Mais comum, permite reattach manual, já temos implementado

**Recomendação**: Manter tmux, mas considerar PTY no futuro se precisarmos de streaming melhor

### 3. **Estrutura de Mensagens Rica**
OpenCode usa uma estrutura muito mais rica para mensagens (parts), o que permite:
- Melhor organização
- Metadata rica
- Suporte a múltiplos tipos de conteúdo

**Recomendação**: Considerar evoluir estrutura de mensagens para suportar parts

### 4. **Storage em Arquivos vs Database**
- **OpenCode (arquivos)**: Mais simples, fácil de debugar, versionável
- **Forge Agent (SQLite)**: Mais estruturado, queries mais fáceis

**Ambos funcionam**, mas arquivos podem ser mais simples para desenvolvimento

### 5. **Client/Server Architecture**
OpenCode usa client/server, permitindo:
- Múltiplos clientes (TUI, Desktop, Web)
- Server roda localmente
- Clientes se conectam via WebSocket/HTTP

**Forge Agent** já tem isso (API REST), mas poderia adicionar WebSocket para streaming

---

## 🔄 Vector DB vs Sessions

### OpenCode NÃO usa Vector DB
- Usa **compaction** (resumos via LLM)
- Usa **summary** (resumos de sessões)
- Usa **pruning** (remove outputs antigos)

### Por que não Vector DB?
1. **Custo**: Embeddings são caros
2. **Latência**: Adiciona latência às queries
3. **Complexidade**: Adiciona infraestrutura
4. **Compaction funciona**: LLM consegue resumir bem o contexto

### Quando Vector DB faz sentido?
- **Codebase muito grande**: Quando precisa buscar em milhões de arquivos
- **Busca semântica**: Quando precisa encontrar código similar
- **RAG**: Quando precisa recuperar contexto relevante de código

**OpenCode usa LSP para busca semântica**, não vector DB.

---

## 🖥️ Gerenciamento de Web + Terminal (PTY)

### Como OpenCode Resolve o Problema

**A chave**: OpenCode **separa completamente** Agent Sessions de PTY Sessions!

#### 1. **Agent Sessions (Session)**
- **Propósito**: Conversas com LLM, execução de tools
- **Storage**: Arquivos JSON em `storage/session/{projectID}/{sessionID}.json`
- **Lifetime**: Persistem até serem deletadas
- **Tools**: Executam comandos via `spawn()` diretamente, **NÃO usam PTY**

```typescript
// opencode/packages/opencode/src/tool/bash.ts
// Tool bash executa comandos diretamente, não via PTY
const proc = spawn(params.command, {
  shell,
  cwd: params.workdir || Instance.directory,  // Sempre especifica cwd
  stdio: ["ignore", "pipe", "pipe"],
})
// Captura stdout/stderr diretamente
```

#### 2. **PTY Sessions (Pty)**
- **Propósito**: Terminais interativos para UI web/desktop
- **Storage**: **Em memória apenas** (não persistem)
- **Lifetime**: Existem apenas enquanto ativas
- **Conexão**: WebSocket para streaming em tempo real
- **Uso**: Apenas para mostrar terminal na UI, **não usado por tools**

```typescript
// opencode/packages/opencode/src/pty/index.ts
interface ActiveSession {
  info: Info
  process: IPty
  buffer: string
  subscribers: Set<WSContext>  // Múltiplos clientes podem se conectar
}

// WebSocket connection
export function connect(id: string, ws: WSContext) {
  session.subscribers.add(ws)  // Adiciona cliente
  // Envia buffer existente
  // Retorna handlers para onMessage/onClose
}
```

#### 3. **Arquitetura de Separação**

```
┌─────────────────────────────────────────────────────────┐
│              OpenCode Server                             │
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │ Agent Sessions   │      │  PTY Sessions    │       │
│  │                  │      │                  │       │
│  │ - Conversas LLM  │      │ - Terminais      │       │
│  │ - Tools exec     │      │ - WebSocket      │       │
│  │ - Storage JSON   │      │ - Em memória      │       │
│  │ - Persistem      │      │ - Não persistem  │       │
│  └──────────────────┘      └──────────────────┘       │
│         │                          │                    │
│         │                          │                    │
│         ▼                          ▼                    │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Tool: bash      │      │  WebSocket API   │       │
│  │  spawn() direto  │      │  /pty/:id/connect│       │
│  │  NÃO usa PTY     │      │                  │       │
│  └──────────────────┘      └──────────────────┘       │
└─────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐      ┌──────────────────┐
│  Web UI          │      │  Web Terminal UI │
│  (Chat)          │      │  (Terminal)      │
│                  │      │                  │
│  - HTTP REST     │      │  - WebSocket     │
│  - Sessions      │      │  - PTY streaming │
└──────────────────┘      └──────────────────┘
```

### Por que essa Separação Funciona?

1. **Tools não precisam de terminal interativo**
   - Tools executam comandos via `spawn()`, capturam output, retornam resultado
   - Não precisam de terminal "vivo" com prompt, histórico, etc.
   - Cada comando é independente e especifica seu próprio `cwd`

2. **PTY é apenas para UX (experiência do usuário)**
   - **Terminal interativo na web**: Usuário pode digitar comandos diretamente, ver output em tempo real
   - **Debug manual**: Usuário pode testar comandos antes de pedir ao agente
   - **Múltiplas abas**: Usuário pode ter vários terminais abertos simultaneamente
   - **Terminal é opcional**: Não é necessário para o agente funcionar, é apenas uma conveniência
   - **Múltiplos clientes**: Vários clientes podem se conectar ao mesmo PTY (colaboração)

3. **Sem conflito de estado**
   - Agent sessions não interferem com PTY sessions
   - Cada um tem seu próprio ciclo de vida
   - PTY pode ser criado/destruído independentemente

### Para que serve o PTY então?

**PTY = Terminal Interativo na Web (como um terminal "normal" no navegador)**

```typescript
// opencode/packages/app/src/components/terminal.tsx
// Usuário pode:
// 1. Digitar comandos diretamente no terminal
// 2. Ver output em tempo real via WebSocket
// 3. Ter múltiplas abas de terminal
// 4. Usar como um terminal "normal" para debug manual

t.onData((data) => {
  // Envia input do usuário para o PTY
  socket.send(data)
})

socket.addEventListener("message", (event) => {
  // Mostra output do PTY no terminal
  t.write(event.data)
})
```

### Estrutura da Interface OpenCode

**A interface tem 3 formas de interação separadas:**

1. **PromptInput (modo normal)**: Chat com o agente
   - Usuário digita mensagens em linguagem natural
   - Agente responde e executa tools automaticamente
   - Tools executam via `spawn()` direto (não PTY)

2. **PromptInput (modo shell)**: Executa comando shell via agente
   - Usuário digita comando (ex: `ls -la`)
   - Chama `session.shell()` que executa via `spawn()` direto
   - **NÃO usa PTY** - apenas executa e mostra resultado no chat

3. **Terminal PTY**: Terminal interativo separado
   - Painel opcional na parte inferior da tela
   - Usuário digita comandos diretamente no terminal
   - **Completamente separado** do agente e do chat
   - Apenas para conveniência do usuário (debug manual, exploração)

**Caso de uso típico:**
```
1. Usuário abre terminal PTY na web (painel inferior)
2. Digita `ls -la` manualmente no terminal para ver arquivos
3. Depois usa PromptInput (chat) e pede: "crie um arquivo novo.txt"
4. Agente executa via tool `bash` (spawn direto, não via PTY)
5. Usuário pode verificar no terminal PTY manualmente depois
```

**Resumo:**
- **Chat com agente (PromptInput)**: Interação via linguagem natural
- **Modo shell (PromptInput)**: Executa comando via `spawn()` (não PTY)
- **Terminal PTY**: Terminal interativo separado, apenas para o usuário
- **Tools do agente**: Executam via `spawn()` direto, não usam PTY
- **PTY**: Apenas para terminal interativo na UI, opcional, para conveniência do usuário

### Comparação com Forge Agent

**Forge Agent (nossa abordagem atual):**
- **Uma sessão tmux por sessão do agente**: Mapeamento 1:1
- **Tools executam no tmux**: Todos comandos via `send_keys`
- **Problema**: Mistura execução de tools com terminal interativo
- **Vantagem**: Contexto persiste automaticamente (cwd, env vars)

**OpenCode (abordagem deles):**
- **Agent sessions independentes de PTY**: Separação completa
- **Tools executam diretamente**: `spawn()` com `cwd` explícito
- **PTY opcional**: Apenas para UI, não usado por tools
- **Vantagem**: Mais simples, sem dependência de terminal

### Como Resolver no Forge Agent?

**Opção 1: Manter Tmux (atual)**
- ✅ Contexto persiste automaticamente
- ✅ Funciona bem para nosso caso
- ❌ Mistura execução com terminal interativo
- ❌ Dependência de tmux

**Opção 2: Separar como OpenCode**
- ✅ Separação clara de responsabilidades
- ✅ Tools mais simples (spawn direto)
- ✅ Terminal opcional para UI
- ❌ Precisa passar `cwd` explicitamente em cada comando
- ❌ Perde persistência automática de contexto

**Opção 3: Híbrido (Recomendado)**
- **Para tools**: Usar `spawn()` direto com `cwd` do tmux (não via send_keys)
- **Para terminal UI**: Criar PTY separado opcional
- **Manter tmux**: Apenas para obter `cwd` atual, não para execução

```python
# Abordagem híbrida
async def execute_command(self, session_name: str, command: str):
    # 1. Obter cwd do tmux
    cwd = await self.get_working_directory(session_name)
    
    # 2. Executar diretamente com spawn (não via send_keys)
    proc = await asyncio.create_subprocess_exec(
        command,
        cwd=cwd,  # Usa cwd do tmux
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    # 3. Capturar output
    stdout, stderr = await proc.communicate()
    
    # 4. Se comando foi 'cd', atualizar tmux também
    if command.startswith('cd '):
        await self.update_tmux_cwd(session_name, new_cwd)
```

**Vantagens da abordagem híbrida:**
- ✅ Execução mais confiável (não depende de capture-pane)
- ✅ Output capturado corretamente
- ✅ Mantém contexto do tmux para cwd
- ✅ Permite adicionar terminal UI opcional depois

---

## 🔧 Arquitetura de Tools e Permissões

### Como OpenCode Estrutura suas Tools

**OpenCode usa um sistema de Registry de Tools similar ao Forge Agent, mas com algumas diferenças importantes:**

#### 1. **Estrutura de Tools**

```typescript
// opencode/packages/opencode/src/tool/registry.ts
// Tools são registradas em um registry central
export namespace ToolRegistry {
  async function all(): Promise<Tool.Info[]> {
    return [
      InvalidTool,
      QuestionTool,      // Apenas para clientes (app/cli/desktop)
      BashTool,         // Executa comandos shell
      ReadTool,         // Lê arquivos
      GlobTool,         // Busca arquivos por padrão
      GrepTool,         // Busca texto em arquivos
      EditTool,         // Edita arquivos (find/replace)
      WriteTool,        // Escreve arquivos
      TaskTool,          // Gerencia tarefas
      WebFetchTool,     // Busca na web
      TodoWriteTool,    // Escreve TODOs
      TodoReadTool,     // Lê TODOs
      WebSearchTool,     // Busca web (exa)
      CodeSearchTool,    // Busca código (exa)
      SkillTool,        // Skills customizados
      LspTool,          // Language Server Protocol
      BatchTool,        // Executa múltiplas tools
      PlanEnterTool,    // Entra em modo plan
      PlanExitTool,     // Sai de modo plan
      ...custom,        // Tools customizadas de plugins
    ]
  }
}
```

**Comparação com Forge Agent:**
- **Forge Agent**: Tools específicas (filesystem, git, github, shell, system, tmux)
- **OpenCode**: Tools mais granulares (read, write, edit, grep, glob, bash, etc.)

#### 2. **Sistema de Permissões**

**OpenCode usa um sistema de permissões baseado em Rulesets:**

```typescript
// opencode/packages/opencode/src/permission/next.ts
export namespace PermissionNext {
  export type Action = "allow" | "deny" | "ask"
  
  export type Rule = {
    permission: string  // Ex: "bash", "edit", "read"
    pattern: string     // Ex: "*", "*.py", "/tmp/*"
    action: Action      // allow, deny, ou ask
  }
  
  export type Ruleset = Rule[]
}
```

**Como funciona:**

1. **Permissões por Tool**: Cada tool tem uma permissão associada
   - `bash` → permissão `"bash"`
   - `edit`, `write`, `patch`, `multiedit` → permissão `"edit"`
   - `read` → permissão `"read"`
   - `grep`, `glob` → permissão `"grep"`

2. **Patterns com Wildcards**: Permissões podem ser específicas por padrão
   ```typescript
   {
     permission: "bash",
     pattern: "rm -rf *",      // Comando específico
     action: "deny"
   }
   {
     permission: "edit",
     pattern: "*.py",           // Arquivos Python
     action: "ask"
   }
   {
     permission: "read",
     pattern: "/tmp/*",         // Diretório específico
     action: "deny"
   }
   ```

3. **Ações:**
   - **`allow`**: Permite automaticamente
   - **`deny`**: Bloqueia automaticamente
   - **`ask`**: Pede confirmação ao usuário

4. **Verificação durante execução:**
   ```typescript
   // opencode/packages/opencode/src/tool/bash.ts
   async execute(params, ctx) {
     // 1. Parse do comando para extrair patterns
     const patterns = extractPatterns(params.command)
     
     // 2. Verifica permissões ANTES de executar
     await ctx.ask({
       permission: "bash",
       patterns: Array.from(patterns),
       always: Array.from(always),
       metadata: {},
     })
     
     // 3. Só executa se permitido
     const proc = spawn(params.command, { ... })
   }
   ```

#### 3. **Sistema de Aprovação (HITL)**

**OpenCode implementa Human-in-the-Loop (HITL) de forma mais sofisticada:**

```typescript
// Quando action === "ask", o sistema:
// 1. Publica evento de permissão pendente
Bus.publish(Event.Asked, {
  id: permissionID,
  sessionID,
  permission: "bash",
  patterns: ["rm -rf *"],
  metadata: { command: "rm -rf /tmp/*" }
})

// 2. Usuário pode responder:
// - "once": Permite apenas esta vez
// - "always": Permite sempre para este pattern
// - "reject": Rejeita e para execução

// 3. Se "always", salva no ruleset para futuras execuções
```

**Comparação com Forge Agent:**
- **Forge Agent**: Aprovação binária (sim/não) por operação
- **OpenCode**: Aprovação com opções (once/always/reject) e persistência de regras

#### 4. **Limitações e Validações**

**OpenCode implementa várias camadas de validação:**

1. **Validação de Schema (Zod)**:
   ```typescript
   parameters: z.object({
     filePath: z.string().describe("The path to the file"),
     offset: z.number().optional(),
     limit: z.number().optional(),
   })
   ```

2. **Validação de Path**:
   ```typescript
   // Verifica se path está dentro do projeto
   await assertExternalDirectory(ctx, filepath)
   ```

3. **Validação de Comando (bash tool)**:
   ```typescript
   // Parse do comando com tree-sitter
   const tree = await parser().parse(params.command)
   // Extrai patterns e diretórios
   // Verifica permissões antes de executar
   ```

4. **Proteção contra Doom Loops**:
   ```typescript
   // Detecta se mesma tool foi chamada 3x com mesmos parâmetros
   if (lastThree.every(p => 
     p.tool === toolName && 
     JSON.stringify(p.input) === JSON.stringify(currentInput)
   )) {
     await PermissionNext.ask({
       permission: "doom_loop",
       patterns: [toolName],
       ...
     })
   }
   ```

#### 5. **Tools Customizadas**

**OpenCode permite tools customizadas via plugins:**

```typescript
// Tools podem ser carregadas de:
// 1. Diretórios do projeto: {tool,tools}/*.{js,ts}
// 2. Plugins instalados
// 3. MCP (Model Context Protocol) servers

const custom = []
for (const dir of await Config.directories()) {
  for await (const match of glob.scan("tool/*.{js,ts}")) {
    const mod = await import(match)
    custom.push(fromPlugin(id, mod))
  }
}
```

### Comparação: OpenCode vs Forge Agent

| Aspecto | OpenCode | Forge Agent |
|---------|----------|-------------|
| **Estrutura** | Tools granulares (read, write, edit, grep) | Tools por domínio (filesystem, git, shell) |
| **Permissões** | Ruleset com patterns e wildcards | Operações específicas (APPROVAL_REQUIRED) |
| **Aprovação** | once/always/reject com persistência | Sim/Não binário |
| **Validação** | Schema + Path + Command parsing | Schema + Path validation |
| **Customização** | Plugins + MCP + arquivos locais | Apenas código |
| **Doom Loop** | ✅ Detecta loops | ❌ Não tem |
| **External Dir** | ✅ Verifica diretórios externos | ❌ Não tem |

### Vantagens da Abordagem OpenCode

1. **Granularidade**: Tools mais específicas permitem controle fino
2. **Flexibilidade**: Patterns com wildcards permitem regras complexas
3. **Persistência**: Regras "always" são salvas automaticamente
4. **Extensibilidade**: Plugins e MCP permitem extensão fácil
5. **Segurança**: Múltiplas camadas de validação

### Vantagens da Abordagem Forge Agent

1. **Simplicidade**: Tools por domínio são mais fáceis de entender
2. **Menos overhead**: Menos tools = menos overhead de registro
3. **Agrupamento lógico**: Operações relacionadas ficam juntas
4. **Mais direto**: Aprovação binária é mais simples

### Recomendações para Forge Agent

**Curto Prazo:**
1. Adicionar detecção de doom loops
2. Melhorar sistema de aprovação (once/always/reject)
3. Adicionar validação de comandos bash (parsing)

**Médio Prazo:**
1. Considerar sistema de rulesets com patterns
2. Adicionar suporte a plugins customizados
3. Implementar persistência de regras de aprovação

**Longo Prazo:**
1. Suporte a MCP (Model Context Protocol)
2. Tools mais granulares se necessário
3. Sistema de wildcards para permissões

---

## 🤖 Integração com LLMs

### Como OpenCode se Comunica com LLMs

**OpenCode usa o Vercel AI SDK (`ai`) com suporte a múltiplos provedores:**

#### 1. **Provedores Suportados**

OpenCode suporta **75+ provedores de LLM** através de:

1. **Provedores Bundled** (incluídos diretamente):
   - OpenAI, Anthropic, Google, Azure, Mistral, Groq
   - DeepInfra, TogetherAI, Perplexity, Vercel
   - Amazon Bedrock, Vertex AI, X.AI, Cohere
   - E mais...

2. **OpenAI-Compatible** (`@ai-sdk/openai-compatible`):
   - Qualquer servidor compatível com API OpenAI
   - **Ollama** (localhost:11434)
   - **LM Studio** (localhost:1234)
   - **LocalAI** (qualquer porta)
   - **Llama.cpp server** (qualquer porta)

3. **Models.dev** (descoberta automática):
   - Busca lista de modelos de `https://models.dev/api.json`
   - Atualiza automaticamente a cada hora
   - Permite descobrir novos modelos sem atualizar código

#### 2. **Configuração de LLMs Locais**

**OpenCode suporta LLMs locais e gratuitas via configuração manual:**

```json
// opencode.json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "llama2": {
          "name": "Llama 2"
        },
        "mistral": {
          "name": "Mistral"
        }
      }
    },
    "lmstudio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LM Studio (local)",
      "options": {
        "baseURL": "http://127.0.0.1:1234/v1"
      },
      "models": {
        "google/gemma-3n-e4b": {
          "name": "Gemma 3n-e4b (local)"
        }
      }
    }
  }
}
```

**Como funciona:**
1. Usa `@ai-sdk/openai-compatible` que aceita qualquer `baseURL`
2. Configura `baseURL` para apontar ao servidor local
3. Define modelos disponíveis manualmente
4. Não precisa de API key (pode usar `"apiKey": "not-needed"`)

#### 3. **Models.dev - Descoberta Automática**

**OpenCode usa `models.dev` para descobrir providers automaticamente:**

```typescript
// opencode/packages/opencode/src/provider/models.ts
export async function get() {
  // 1. Tenta ler cache local
  const file = Bun.file(filepath)
  const result = await file.json().catch(() => {})
  if (result) return result
  
  // 2. Busca de models.dev
  const json = await fetch("https://models.dev/api.json").then((x) => x.text())
  return JSON.parse(json)
}

// Atualiza automaticamente a cada hora
setInterval(() => ModelsDev.refresh(), 60 * 1000 * 60)
```

**Vantagens:**
- Descobre novos modelos automaticamente
- Não precisa atualizar código para novos providers
- Cache local para funcionar offline
- Metadados completos (limites, custos, capacidades)

#### 4. **Sistema de Provedores**

**OpenCode tem um sistema flexível de provedores:**

```typescript
// 1. Provedores podem vir de:
// - models.dev (descoberta automática)
// - Config manual (opencode.json)
// - Plugins customizados

// 2. Cada provedor pode ter:
{
  id: "ollama",
  name: "Ollama (local)",
  npm: "@ai-sdk/openai-compatible",  // Package a usar
  options: {
    baseURL: "http://localhost:11434/v1",
    apiKey: "not-needed",  // Opcional
    timeout: 300000,       // Opcional
  },
  models: {
    "llama2": {
      name: "Llama 2",
      limit: { context: 4096, output: 2048 },
      capabilities: { tool_call: true, temperature: true }
    }
  }
}

// 3. SDK é criado dinamicamente:
const sdk = createOpenAICompatible({
  baseURL: "http://localhost:11434/v1"
})
const model = sdk.languageModel("llama2")
```

### Comparação: OpenCode vs Forge Agent

| Aspecto | OpenCode | Forge Agent |
|---------|----------|-------------|
| **Provedores** | 75+ via models.dev + config | Ollama, LocalAI, AirLLM (manual) |
| **LLMs Locais** | ✅ Via `openai-compatible` + config | ✅ Suporte nativo (Ollama, LocalAI) |
| **Descoberta** | ✅ Automática (models.dev) | ❌ Manual (config) |
| **Configuração** | JSON (opencode.json) | YAML (agent.yaml) |
| **SDK** | Vercel AI SDK (`ai`) | Implementação própria |
| **Gratuitas** | ✅ Suporta (Ollama, LM Studio) | ✅ Suporta (Ollama, LocalAI) |

### Vantagens da Abordagem OpenCode

1. **Flexibilidade**: Qualquer servidor OpenAI-compatible funciona
2. **Descoberta Automática**: Models.dev atualiza automaticamente
3. **Padrão**: Usa padrão da indústria (Vercel AI SDK)
4. **Extensibilidade**: Fácil adicionar novos providers via config

### Vantagens da Abordagem Forge Agent

1. **Simplicidade**: Configuração mais direta (YAML)
2. **Foco Local**: Otimizado para LLMs locais
3. **Controle**: Implementação própria = mais controle
4. **Menos Dependências**: Não depende de models.dev

### Como Forge Agent Pode Melhorar

**Curto Prazo:**
1. Adicionar suporte a `openai-compatible` genérico
2. Permitir configurar `baseURL` customizado
3. Suportar múltiplos provedores via config

**Médio Prazo:**
1. Considerar usar Vercel AI SDK (padrão da indústria)
2. Adicionar descoberta automática de modelos (opcional)
3. Suportar models.dev ou similar

**Longo Prazo:**
1. Sistema de plugins para providers customizados
2. Cache de metadados de modelos
3. UI para configurar providers facilmente

---

## 🚀 Fluxo de Execução e Auto-Correção

### Por que OpenCode Parece Mais "Fluido"?

**A diferença principal está no sistema de execução automática e loop contínuo:**

#### 1. **Execução Automática Durante Streaming**

**OpenCode usa Vercel AI SDK que executa tools automaticamente:**

```typescript
// opencode/packages/opencode/src/session/llm.ts
return streamText({
  tools,  // Tools são passadas para o SDK
  // SDK automaticamente executa tools quando LLM chama
  // Não para para aprovação a cada tool
})

// Durante o streaming:
// 1. LLM chama tool → SDK executa imediatamente
// 2. Resultado volta para LLM → LLM continua pensando
// 3. LLM pode auto-corrigir baseado no resultado
// 4. Loop continua até LLM terminar naturalmente
```

**Comparação com Forge Agent:**
- **Forge Agent**: Para a cada step, espera aprovação, executa, para novamente
- **OpenCode**: Executa automaticamente, LLM vê resultado, continua pensando

#### 2. **Loop Contínuo (Não Para a Cada Step)**

**OpenCode tem um loop que continua até a LLM terminar:**

```typescript
// opencode/packages/opencode/src/session/prompt.ts
export const loop = async (sessionID) => {
  while (true) {  // Loop continua até LLM terminar
    // 1. LLM gera resposta (pode incluir tool calls)
    const result = await processor.process({ ... })
    
    // 2. Se LLM chamou tools, executa automaticamente
    // 3. Resultados voltam para LLM
    // 4. LLM pode continuar pensando/corrigindo
    // 5. Loop continua até LLM terminar naturalmente
    
    if (result === "stop") break  // Só para se erro ou usuário rejeitar
    continue  // Continua o loop
  }
}
```

**Características:**
- **Múltiplos steps**: LLM pode fazer vários steps sem parar
- **Auto-correção**: LLM vê resultados e pode corrigir imediatamente
- **Paralelismo**: LLM pode chamar múltiplas tools em paralelo
- **Continuidade**: Não para entre steps

#### 3. **Aprovações Assíncronas (Não Bloqueiam)**

**OpenCode usa aprovações assíncronas que não bloqueiam o loop:**

```typescript
// opencode/packages/opencode/src/session/processor.ts
case "tool-call": {
  // Tool é chamada → Executa automaticamente
  // Se precisa aprovação, pede assincronamente
  await ctx.ask({
    permission: "bash",
    patterns: ["*"],
  })
  // Se usuário rejeitar, para o loop
  // Se permitir, continua automaticamente
}
```

**Como funciona:**
1. Tool é executada automaticamente
2. Se precisa aprovação, pede em background
3. Se usuário rejeitar → loop para
4. Se permitir → continua automaticamente
5. **Não bloqueia cada tool individualmente**

#### 4. **Instruções para Auto-Correção**

**OpenCode instrui a LLM a auto-corrigir:**

```txt
// opencode/packages/opencode/src/session/prompt/anthropic.txt
- You can call multiple tools in a single response
- Maximize use of parallel tool calls where possible
- If a tool call fails, try a different approach
- Keep going until the problem is solved
```

```txt
// opencode/packages/opencode/src/session/prompt/beast.txt
You MUST iterate and keep going until the problem is solved.
You have everything you need to resolve this problem autonomously.
Only terminate when you are sure the problem is solved.
```

#### 5. **Sistema de Steps (Não Limita Rigidamente)**

**OpenCode tem limite de steps, mas é flexível:**

```typescript
const maxSteps = agent.steps ?? Infinity  // Padrão: infinito
const isLastStep = step >= maxSteps

// Se último step, desabilita tools mas permite texto
if (isLastStep) {
  messages.push({
    role: "assistant",
    content: MAX_STEPS  // "Tools disabled, respond with text only"
  })
}
```

**Vantagens:**
- LLM pode fazer muitos steps antes de parar
- No último step, ainda pode responder em texto
- Não corta abruptamente

### Comparação: OpenCode vs Forge Agent

| Aspecto | OpenCode | Forge Agent |
|---------|----------|-------------|
| **Execução** | Automática durante streaming | Manual, para a cada step |
| **Loop** | Contínuo até LLM terminar | Para após cada step |
| **Auto-correção** | ✅ LLM vê resultado e corrige | ❌ Precisa aprovar cada step |
| **Paralelismo** | ✅ Múltiplas tools em paralelo | ❌ Sequencial |
| **Aprovação** | Assíncrona (não bloqueia) | Síncrona (bloqueia) |
| **Continuidade** | ✅ Continua até resolver | ❌ Para após cada ação |
| **Fluidez** | ✅ Muito fluido | ❌ Travado |

### Por que Forge Agent Parece Travado?

**Problemas na abordagem atual:**

1. **Para a cada step**: Precisa aprovar cada ação individualmente
2. **Sem auto-correção**: LLM não vê resultado e não pode corrigir
3. **Sem paralelismo**: Executa uma tool por vez
4. **Sem continuidade**: Para após cada step, não continua automaticamente
5. **Aprovação bloqueante**: Espera aprovação antes de continuar

### Como Melhorar o Forge Agent

**Curto Prazo:**
1. **Executar tools automaticamente**: Não parar para aprovação a cada tool
2. **Loop contínuo**: Permitir múltiplos steps sem parar
3. **Aprovação assíncrona**: Não bloquear execução
4. **Paralelismo**: Executar múltiplas tools em paralelo

**Médio Prazo:**
1. **Usar Vercel AI SDK**: Para execução automática de tools
2. **Sistema de steps flexível**: Limite alto, não cortar abruptamente
3. **Instruções para auto-correção**: Instruir LLM a continuar até resolver

**Longo Prazo:**
1. **Streaming de resultados**: Mostrar resultados em tempo real
2. **Auto-correção inteligente**: Detectar erros e auto-corrigir
3. **Planejamento adaptativo**: Ajustar plano baseado em resultados

### Exemplo de Fluxo OpenCode

```
1. Usuário: "Crie um arquivo novo.txt"
2. LLM: [Chama WriteTool] → Executa automaticamente
3. Resultado: "Arquivo criado" → Volta para LLM
4. LLM: [Vê resultado, continua] → "Arquivo criado com sucesso"
5. LLM: [Termina naturalmente] → Loop para
```

### Exemplo de Fluxo Forge Agent (Atual)

```
1. Usuário: "Crie um arquivo novo.txt"
2. LLM: [Gera plano com step] → Para e espera
3. Usuário: [Aprova] → Executa
4. Resultado: "Arquivo criado" → Para novamente
5. LLM: [Não vê resultado ainda] → Precisa novo step
6. Usuário: [Aprova novo step] → LLM finalmente vê resultado
```

**Problema**: Muito mais lento, sem auto-correção, travado.

---

## 📦 Vercel AI SDK - O que é e Como Funciona

### O que é o Vercel AI SDK?

**Vercel AI SDK é um toolkit TypeScript/JavaScript para construir apps com LLMs:**

- **Provider-agnóstico**: Funciona com OpenAI, Anthropic, Google, Ollama, etc.
- **Full-stack**: Funciona no backend (Node.js/Edge) e frontend (React/Vue/Svelte)
- **Streaming nativo**: Suporte a streaming de respostas
- **Tool calling automático**: Executa tools automaticamente durante streaming
- **Structured outputs**: Suporte a outputs estruturados

### Arquitetura: Backend vs Frontend

**Backend (Node.js/Edge/Serverless):**
```typescript
// Backend: Rota API que processa LLM
import { openai } from '@ai-sdk/openai'
import { streamText } from 'ai'

export async function POST(req) {
  const { messages } = await req.json()
  
  const result = streamText({
    model: openai('gpt-4'),
    messages,
    tools: {
      readFile: tool({
        description: 'Read a file',
        execute: async ({ path }) => {
          // Executa automaticamente quando LLM chama
          return { content: await fs.readFile(path) }
        }
      })
    }
  })
  
  // Retorna stream para frontend
  return result.toDataStreamResponse()
}
```

**Frontend (React/Vue/Svelte):**
```typescript
// Frontend: Hook React para UI
import { useChat } from '@ai-sdk/react'

function Chat() {
  const { messages, input, handleSubmit } = useChat({
    api: '/api/chat'  // Rota backend acima
  })
  
  return (
    <div>
      {messages.map(m => <div>{m.content}</div>)}
      <form onSubmit={handleSubmit}>
        <input value={input} />
      </form>
    </div>
  )
}
```

### Como Funciona a Execução Automática de Tools?

**O Vercel AI SDK executa tools automaticamente durante o streaming:**

```typescript
const result = streamText({
  model: openai('gpt-4'),
  messages,
  tools: {
    readFile: tool({
      description: 'Read a file',
      execute: async ({ path }) => {
        // Esta função é chamada AUTOMATICAMENTE
        // quando a LLM chama a tool durante o streaming
        return { content: await fs.readFile(path) }
      }
    })
  }
})

// Durante o streaming:
// 1. LLM gera resposta e chama readFile
// 2. SDK automaticamente executa execute()
// 3. Resultado volta para LLM
// 4. LLM continua pensando/corrigindo
// 5. Stream continua até LLM terminar
```

**Características:**
- ✅ **Automático**: Não precisa aprovar cada tool
- ✅ **Durante streaming**: Executa enquanto LLM gera resposta
- ✅ **Resultado imediato**: LLM vê resultado e pode corrigir
- ✅ **Loop contínuo**: Continua até LLM terminar naturalmente

### Suporte a Python?

**❌ Não há SDK oficial para Python**

**Alternativas para Python:**

1. **AI Gateway da Vercel** (recomendado):
   - Gateway que roteia para múltiplos providers
   - Usa SDKs oficiais (OpenAI, Anthropic) em Python
   - Suporta streaming e tool calling

2. **Ports da comunidade**:
   - `python-ai-sdk` (não oficial)
   - Implementações similares em Python

3. **SDKs nativos**:
   - OpenAI SDK Python
   - Anthropic SDK Python
   - Mas sem a abstração unificada do AI SDK

### Suporte a React?

**✅ Suporte oficial completo**

```bash
npm install @ai-sdk/react
```

**Hooks disponíveis:**
- `useChat`: Para interfaces de chat
- `useCompletion`: Para completions simples
- `useAssistant`: Para assistentes com tools

### Suporte a LLMs Locais/Gratuitas?

**✅ Suportado via OpenAI-compatible**

```typescript
import { createOpenAI } from '@ai-sdk/openai'

const ollama = createOpenAI({
  baseURL: 'http://localhost:11434/v1',
  apiKey: 'not-needed'  // Ollama não precisa de key
})

const result = streamText({
  model: ollama('llama3.2'),
  messages,
  tools: { ... }
})
```

**Funciona com:**
- Ollama (localhost:11434)
- LM Studio (localhost:1234)
- LocalAI (qualquer porta)
- Qualquer servidor OpenAI-compatible

### Comparação: Vercel AI SDK vs Implementação Própria

| Aspecto | Vercel AI SDK | Implementação Própria |
|---------|---------------|----------------------|
| **Execução automática** | ✅ Automática durante streaming | ❌ Precisa implementar |
| **Tool calling** | ✅ Nativo, automático | ❌ Precisa implementar |
| **Streaming** | ✅ Nativo | ❌ Precisa implementar |
| **Provider-agnóstico** | ✅ Suporta 20+ providers | ❌ Precisa integrar cada um |
| **Python** | ❌ Não tem | ✅ Pode fazer em Python |
| **React** | ✅ Hooks prontos | ❌ Precisa fazer do zero |
| **Complexidade** | ✅ Baixa (usa lib) | ❌ Alta (implementa tudo) |

### Para o Forge Agent (Python + React)

**Opções:**

1. **Manter implementação própria** (atual):
   - ✅ Controle total
   - ✅ Já funciona
   - ❌ Precisa implementar execução automática
   - ❌ Precisa implementar streaming melhor

2. **Usar AI Gateway + SDKs nativos**:
   - ✅ Abstração de providers
   - ✅ Streaming nativo
   - ❌ Ainda precisa implementar tool execution
   - ❌ Mais complexo

3. **Híbrido: Backend Python + Frontend React com AI SDK**:
   - Backend Python: Processa LLM (OpenAI SDK, etc.)
   - Frontend React: Usa `@ai-sdk/react` para UI
   - Bridge: API REST entre eles
   - ✅ Melhor UX no frontend
   - ✅ Mantém backend Python
   - ❌ Precisa bridge entre Python e React

4. **Migrar backend para TypeScript/Node.js**:
   - ✅ Usa Vercel AI SDK completo
   - ✅ Execução automática nativa
   - ✅ Streaming nativo
   - ❌ Precisa reescrever backend

### Recomendação para Forge Agent

**Curto Prazo:**
- Manter Python backend
- Implementar execução automática de tools (inspirado no AI SDK)
- Melhorar streaming

**Médio Prazo:**
- Considerar usar `@ai-sdk/react` no frontend
- Backend Python pode expor API compatível com AI SDK

**Longo Prazo:**
- Avaliar migração para TypeScript/Node.js se necessário
- Ou criar port do AI SDK para Python (comunidade)

---

## 📊 Comparação Final

| Aspecto | OpenCode | Forge Agent |
|---------|----------|-------------|
| **Interface** | TUI + Desktop + Web | Web only |
| **Execução** | PTY (bun-pty) | Tmux |
| **Storage** | Arquivos JSON | SQLite |
| **Memória** | Compaction + Summary | Todas mensagens |
| **Vector DB** | ❌ Não usa | ❌ Não usa (ainda) |
| **Arquitetura** | Client/Server local | API REST |
| **Tools** | Registry rico | Tools específicos |
| **LSP** | ✅ Integrado | ❌ Não tem |
| **Snapshot** | ✅ Git diffs | ❌ Não tem |
| **Streaming** | ✅ WebSocket | ❌ HTTP only |

---

## 🎯 Recomendações para Forge Agent

### Curto Prazo
1. **Implementar Compaction**: Similar ao OpenCode
2. **Melhorar estrutura de mensagens**: Suportar parts
3. **Adicionar WebSocket**: Para streaming de output

### Médio Prazo
1. **Considerar PTY**: Se precisar de melhor streaming
2. **Adicionar LSP**: Para busca semântica de código
3. **Implementar Snapshot**: Para tracking de mudanças

### Longo Prazo
1. **Vector DB opcional**: Para codebases muito grandes
2. **Múltiplos clientes**: Desktop app, mobile app
3. **Skills system**: Similar ao OpenCode

---

## 🏁 Conclusão

OpenCode e Forge Agent são projetos similares com abordagens diferentes:

- **OpenCode**: Terminal-first, focado em desenvolvedores, usa PTY, compaction automática
- **Forge Agent**: Web-first, focado em acessibilidade, usa tmux, sem compaction ainda

**Principais aprendizados:**
1. Compaction é essencial para sessões longas
2. PTY vs Tmux são trade-offs válidos
3. Estrutura rica de mensagens ajuda muito
4. Vector DB não é necessário para a maioria dos casos
5. Client/Server permite múltiplos clientes

**Recomendação principal**: Implementar compaction similar ao OpenCode para resolver problemas de contexto em sessões longas.
