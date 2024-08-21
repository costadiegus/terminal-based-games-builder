from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeWebsiteTool  # , DallETool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler("detalhamento_execucao.log")
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

# Testando o logger manualmente
logger.debug("Logger configurado corretamente e pronto para uso.")

# LLMs

# Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError(
        "A chave de API do Google não foi fornecida. Verifique o arquivo .env."
    )
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=google_api_key,
)

# Llama
grog_api_key = os.getenv("GROQ_API_KEY")
if not grog_api_key:
    raise ValueError(
        "A chave de API do GROG não foi fornecida. Verifique o arquivo .env."
    )
llama_3 = ChatGroq(
    api_key=grog_api_key,
    model="llama3-70b-8192",  # "llama-3.1-70b-versatile"
    timeout=180,
)

# ChatGPT
gpt_4o = ChatOpenAI(model_name="gpt-4o")
gpt_4o_mini = ChatOpenAI(model_name="gpt-4o-mini")

# Ferramenta de pesquisa na internet
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Ferramenta DALL-E com especificação de tamanho 16:9
# dalle_tool = DallETool(size="1024x1024")  # Formato 16:9

# 1. Pesquisador de Design de Jogos
pesquisador_design_jogos = Agent(
    role="Pesquisador de Design de Jogos",
    goal="Realizar uma pesquisa detalhada sobre design de jogos, incluindo mecânicas, dinâmicas, e componentes-chave do jogo {jogo}.",
    backstory=(
        "Você é um pesquisador especializado em design de jogos, com uma "
        "paixão por explorar e entender as nuances que fazem um jogo ser divertido e envolvente."
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],  # Utiliza ferramentas de pesquisa e LLM
    llm=llama_3,  # Liga o LLM diretamente ao agente
    allow_delegation=False,
)

tarefa_pesquisa_jogo = Task(
    description=(
        "Pesquise informações detalhadas e abrangentes sobre o design do jogo {jogo}. "
        "Foque em aspectos como mecânicas de jogo, dinâmica de grupo, fluxo de jogo, "
        "regras, e componentes visuais. Sua pesquisa deve ser compilada em um documento Markdown."
    ),
    expected_output="Um documento em Markdown com a pesquisa detalhada sobre o design do jogo {jogo}.",
    tools=[search_tool],  # Ferramentas usadas para a pesquisa
    agent=pesquisador_design_jogos,
    output_file="pesquisa_design_jogo.md",  # O nome do arquivo de saída
)

# 2. Pesquisador de Interface de Usuário (UI) para Jogos de Terminal
pesquisador_ui_terminal = Agent(
    role="Pesquisador de UI para Jogos de Terminal",
    goal=(
        "Realizar uma proposta detalhada de Exibição de Estado, Comandos do Usuário e Feedback ao Jogador "
        "para a versão de terminal do jogo {jogo}."
    ),
    backstory=(
        "Você é um especialista em design de interfaces de usuário, focado em criar experiências "
        "intuitivas e agradáveis para jogos de terminal. Seu trabalho é garantir que os jogadores "
        "tenham uma interface clara e fácil de usar enquanto jogam no terminal."
    ),
    verbose=True,
    memory=True,
    tools=[search_tool],  # Utiliza ferramentas de pesquisa e LLM
    llm=llama_3,  # Liga o LLM diretamente ao agente
    allow_delegation=False,
)

tarefa_proposta_ui = Task(
    description=(
        "Com base na pesquisa realizada pelo agente 'pesquisador_design_jogos' sobre o jogo {jogo}, "
        "desenvolva uma proposta detalhada de Interface de Usuário para a versão de terminal do jogo. "
        "A proposta deve incluir seções sobre Exibição de Estado, Comandos do Usuário e Feedback ao Jogador. "
        "O objetivo é orientar o desenvolvimento de uma versão de terminal do jogo informado."
    ),
    expected_output=(
        "Um documento em Markdown com a proposta detalhada de Interface de Usuário, "
        "incluindo Exibição de Estado, Comandos do Usuário e Feedback ao Jogador."
    ),
    tools=[search_tool],  # Ferramenta usada para gerar a proposta baseada na pesquisa
    agent=pesquisador_ui_terminal,
    output_file="proposta_ui_terminal.md",  # O nome do arquivo de saída
)

# 3. Engenheiro de Software para Jogos de Terminal
engenheiro_software_terminal = Agent(
    role="Engenheiro de Software para Jogos de Terminal",
    goal=(
        "Desenvolver uma documentação detalhada sobre a estrutura de código para a versão de terminal do jogo {jogo}, "
        "incluindo organização em classes, lógica de jogo, e gerenciamento de estado."
    ),
    backstory=(
        "Você é um engenheiro de software especializado em desenvolvimento de jogos para terminal, com vasta experiência "
        "em criar arquiteturas de código eficientes e escaláveis para jogos baseados em linha de comando."
    ),
    verbose=True,
    memory=True,
    # tools=[search_tool],  # Utiliza ferramentas de pesquisa e LLM
    llm=llama_3,  # Liga o LLM diretamente ao agente
    allow_delegation=False,
)

tarefa_documentacao_codigo = Task(
    description=(
        "Com base na pesquisa realizada pelo agente 'pesquisador_design_jogos' e na proposta realizada pelo 'pesquisador_ui_terminal' "
        "sobre o jogo {jogo}, desenvolva uma documentação detalhada em Markdown abordando a estrutura de código recomendada. "
        "A documentação deve cobrir organização em classes (ex.: `Carta`, `Baralho`, `Jogador`, `Tabuleiro`), lógica de jogo "
        "(regras, verificações, turnos), e gerenciamento de estado (estado atual do jogo, controle de turnos, etc.)."
    ),
    expected_output=(
        "Um documento em Markdown com a documentação detalhada sobre a estrutura de código para a versão de terminal do jogo {jogo}."
    ),
    # tools=[search_tool],  # Ferramenta usada para gerar a documentação
    agent=engenheiro_software_terminal,
    output_file="documentacao_estrutura_codigo.md",  # O nome do arquivo de saída
)

# 4. Agente Criador de Prompts para DALL-E
criador_prompt_dalle = Agent(
    role="Criador de Prompts para DALL-E",
    goal="Escrever um prompt para gerar uma imagem usando DALL-E com base no jogo {jogo} em sua versão de terminal",
    verbose=True,
    memory=True,
    backstory=(
        "Você é um especialista em criar descrições detalhadas e imaginativas que "
        "permitem ao DALL-E gerar imagens impressionantes com base em textos."
    ),
    llm=llama_3,
)

tarefa_criacao_prompt_dalle = Task(
    description=(
        "Criar um prompt detalhado para gerar uma imagem criativa no DALL-E "
        "com base no jogo {jogo} destacando sua versão de terminal"
    ),
    expected_output="Um prompt de texto detalhado para geração de imagem no DALL-E.",
    agent=criador_prompt_dalle,
)

# 5. Agente Gerador de Imagens com DALL-E
gerador_imagens = Agent(
    role="Gerador de Imagens com DALL-E",
    goal="Gerar uma imagem usando DALL-E com o prompt fornecido pelo Criador de Prompts para DALL-E",
    verbose=True,
    memory=True,
    backstory=(
        "Você é um mestre em transformar descrições textuais em belas imagens, utilizando o poder do DALL-E."
    ),
    # tools=[dalle_tool],
    llm=gpt_4o_mini,
    allow_delegation=False,
)

tarefa_geracao_imagem = Task(
    description=(
        "Gerar uma imagem usando o DALL-E com o prompt fornecido pelo Criador de Prompts para DALL-E."
    ),
    expected_output="Uma imagem gerada pronta para uso.",
    agent=gerador_imagens,
)

# 6. Agente Revisor
revisor = Agent(
    role="Revisor de Conteúdo",
    goal="Revisar todo o conteúdo produzido, incluir links das imagens geradas e entregar a versão final ao usuário",
    verbose=True,
    memory=True,
    backstory=(
        "Você tem um olho afiado para detalhes, garantindo que todo o conteúdo "
        "esteja perfeito antes de ser entregue ao usuário."
    ),
    llm=gpt_4o_mini,
    allow_delegation=False,
)

tarefa_revisao = Task(
    description=(
        "Revisar todo o conteúdo produzido (pesquisa, UI e documentação de código), "
        "incluir os links das imagens geradas e preparar a versão final para entrega ao usuário."
        "Todo o texto deve estar em Português Brasil."
    ),
    expected_output="Conteúdo revisado, com links das imagens inclusos, pronto para entrega ao usuário.",
    agent=revisor,
    output_file="especificacao_jogo.md",  # Configurando o output para salvar em um arquivo Markdown
)

# Formando a crew
crew = Crew(
    agents=[
        pesquisador_design_jogos,
        pesquisador_ui_terminal,
        engenheiro_software_terminal,
        # criador_prompt_dalle,
        # gerador_imagens,
        # revisor,
    ],
    tasks=[
        tarefa_pesquisa_jogo,
        tarefa_proposta_ui,
        tarefa_documentacao_codigo,
        # tarefa_criacao_prompt_dalle,
        # tarefa_geracao_imagem,
        # tarefa_revisao,
    ],
    verbose=True,
    logger=logger,
    manager_llm=llama_3,
    function_calling_llm=llama_3,
    max_rpm=30,
    process=Process.sequential,  # Processamento sequencial das tarefas
)

jogo = input("Digite o nome do jogo: ")

print(jogo)

# Inicia a execução da tarefa
result = crew.kickoff(inputs={"jogo": jogo})  # Exemplo com o jogo UNO
logger.info(result)
print("Execução detalhada salva em detalhamento_execucao.log")
