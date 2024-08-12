from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# LLM
grog_api_key = os.getenv("GROQ_API_KEY")
if not grog_api_key:
    raise ValueError(
        "A chave de API do GROG não foi fornecida. Verifique o arquivo .env."
    )

llama3 = ChatGroq(
    api_key=grog_api_key,
    model="llama3-70b-8192",  # "llama-3.1-70b-versatile"
    timeout=180,
)

gpt4o = ChatOpenAI(model_name="gpt-4o")

gpt4omini = ChatOpenAI(model_name="gpt-4o-mini")


# Função para salvar conteúdo em um arquivo markdown
@tool("save_markdown")
def save_markdown(content: str, filename: str):
    """
    Tool para salvar o conteúdo em um arquivo markdown.
    """
    try:
        with open(
            filename, "w", encoding="utf-8"
        ) as file:  # Adiciona a codificação UTF-8
            file.write(content)
        return f"Conteúdo salvo em {filename}."
    except Exception as e:
        return f"Erro ao salvar conteúdo em {filename}: {e}"


# Nome do jogo
game_name = input("Qual o jogo que gostaria de construir na versão para terminal?\n")
if not game_name:
    print("O nome do jogo não pode estar vazio.")
    exit(1)

# Definição dos agentes
designer = Agent(
    role="Designer Mecânicas",
    goal=f'Criar as mecânicas de jogo, regras e dinâmica para o jogo "{game_name}", adaptando-as para a jogabilidade em um ambiente de terminal.',
    backstory=f"""
        Com vasta experiência em adaptar jogos complexos para plataformas simples, 
        você é especialista em transformar jogos tradicionais, como '{game_name}', 
        em versões jogáveis em um terminal, focando em jogabilidade fluida e envolvente 
        dentro das limitações do texto.
    """,
    verbose=True,
    memory=True,
    llm=llama3,
)

narrador = Agent(
    role="Narrador de Terminal",
    goal=f'Desenvolver a narrativa e os diálogos do jogo "{game_name}", otimizados para uma experiência textual em terminal.',
    backstory=f"""
        Você é um mestre em contar histórias que capturam a imaginação dos jogadores usando apenas texto. 
        Com habilidade em criar descrições vívidas e diálogos envolventes, seu foco é garantir que a narrativa de '{game_name}' 
        se destaque mesmo em uma interface de terminal.
    """,
    verbose=True,
    memory=True,
    llm=llama3,
)

engenheiro = Agent(
    role="Engenheiro de Software",
    goal=f'Garantir que a especificação do jogo "{game_name}" seja tecnicamente viável para implementação em um ambiente de terminal, focando na usabilidade e nas limitações do terminal.',
    backstory=f"""
        Com expertise em desenvolver jogos para ambientes de linha de comando, você se destaca em criar experiências jogáveis 
        eficientes e divertidas, mesmo com os recursos limitados de uma interface de texto, especialmente para o jogo '{game_name}'.
    """,
    verbose=True,
    memory=True,
    llm=llama3,
)

especialista_ux = Agent(
    role="Especialista UX",
    goal=f'Assegurar que a interface e a experiência do jogador em um terminal sejam intuitivas e agradáveis no jogo "{game_name}".',
    backstory=f"""
        Você é um designer de UX que entende profundamente como criar interações de usuário fluidas e naturais em interfaces baseadas em texto. 
        Seu foco é garantir que o jogo '{game_name}' seja fácil de usar, mesmo em um terminal.
    """,
    verbose=True,
    memory=True,
    llm=llama3,
)

documentarista = Agent(
    role="Documentarista Técnico",
    goal=f'Criar uma documentação detalhada do jogo "{game_name}", cobrindo todos os aspectos técnicos e de design, com foco na implementação em um ambiente de terminal.',
    backstory=f"""
        Com um olhar atento para detalhes, você é responsável por documentar todos os elementos do jogo '{game_name}', 
        assegurando que os desenvolvedores tenham todas as informações necessárias para implementar o jogo de forma precisa em um terminal.
    """,
    verbose=True,
    memory=True,
    llm=llama3,
    allow_delegation=False,
)

# Definição das tasks associadas às suas respectivas tools
design_mechanics_task = Task(
    description=f"""
        Desenvolver as mecânicas e regras do jogo "{game_name}", adaptadas para a jogabilidade em um ambiente de terminal.
        Certifique-se de incluir todas as regras principais, condições de vitória, e mecânicas especiais. 
        O resultado deve ser um documento markdown detalhado.
    """,
    expected_output="mechanics.md",
    agent=designer,
    tools=[save_markdown],
    tool_args={"filename": "mechanics.md"},
)

write_narrative_task = Task(
    description=f"""
        Criar a narrativa, diálogos, e descrições do jogo "{game_name}", otimizados para uma experiência textual em terminal.
        A narrativa deve ser envolvente e adequada para o ambiente de terminal.
        O resultado deve ser um documento markdown detalhado.
    """,
    expected_output="narrative.md",
    agent=narrador,
    tools=[save_markdown],
    tool_args={"filename": "narrative.md"},
)

develop_technical_specifications_task = Task(
    description=f"""
        Elaborar as especificações técnicas do jogo "{game_name}", detalhando a estrutura do código, lógica do jogo,
        e como cada mecânica será implementada em um ambiente de terminal. 
        Incluir diagramas de fluxo e pseudocódigo, se necessário. 
        O resultado deve ser um documento markdown detalhado.
    """,
    expected_output="technical_specifications.md",
    agent=engenheiro,
    tools=[save_markdown],
    tool_args={"filename": "technical_specifications.md"},
)

design_ux_task = Task(
    description=f"""
        Especificar a experiência do usuário (UX) para o jogo "{game_name}" no terminal.
        Detalhar como o usuário interagirá com o jogo, incluindo comandos, interface de texto, e fluxo de navegação.
        O resultado deve ser um documento markdown detalhado.
    """,
    expected_output="ux_design.md",
    agent=especialista_ux,
    tools=[save_markdown],
    tool_args={"filename": "ux_design.md"},
)

compile_final_documentation_task = Task(
    description=f"""
        Compilar todos os documentos gerados (mecânicas, narrativa, especificações técnicas, UX) em uma documentação final completa do jogo "{game_name}".
        Assegurar que todos os documentos estejam bem formatados e organizados.
        O resultado deve ser um documento markdown chamado 'final_specification.md' no diretório especificado.
    """,
    expected_output="final_specification.md",
    agent=documentarista,
    tools=[save_markdown],
    tool_args={"filename": "final_specification.md"},
    allow_delegation=False,
)

# Criação da Crew
specification_crew = Crew(
    agents=[designer, narrador, engenheiro, especialista_ux, documentarista],
    tasks=[
        design_mechanics_task,
        write_narrative_task,
        develop_technical_specifications_task,
        design_ux_task,
        compile_final_documentation_task,
    ],
    process=Process.sequential,
    verbose=True,
    memory=True,
    manager_llm=llama3,
    function_calling_llm=llama3,
    allow_delegation=False,
    async_execution=False,
    max_rpm=30,
)

# Executa a Crew
try:
    result = specification_crew.kickoff(inputs={"game_name": game_name})
    # Salvando o conteúdo de 'result' em um arquivo .log
    log_file_path = "execution.log"
    try:
        with open(log_file_path, "w") as log_file:
            log_file.write(
                str(result)
            )  # Convertendo 'result' para string e escrevendo no arquivo
        print(f"Log salvo em: {log_file_path}")
    except Exception as e:
        print(f"Erro ao salvar o log: {e}")
except Exception as e:
    print(f"Erro durante a execução da Crew: {e}")
    result = None

# Print results
print("\n\n########################")
print("## FIM DA EXECUÇÃO")
