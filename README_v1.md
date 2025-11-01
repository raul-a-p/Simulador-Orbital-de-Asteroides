╔═══════════════════════════════════════════════════════════════════╗
║       SIMULADOR ORBITAL DE ASTEROIDES V1 - GUIA DE USO            ║
╚═══════════════════════════════════════════════════════════════════╝

📚 CLASSES PRINCIPAIS:

1. CorpoCeleste
   - Representa um corpo celeste (Sol, Terra, Asteroide)
   - Atributos: nome, massa, posicao, velocidade, cor
   - Métodos: salvar_estado(), energia_cinetica(), momento_angular()

2. SistemaGravitacional
   - Gerencia múltiplos corpos e evolução temporal
   - Métodos principais:
     * adicionar_corpo(corpo)
     * simular(tempo_total, progresso=True)
     * energia_total()
     * momento_angular_total()

3. ResultadoSimulacao
   - Armazena resultados da simulação
   - Método: gerar_relatorio()

⚙️  FUNÇÕES DE CONFIGURAÇÃO:

- criar_sistema_terra_sol()      → Sistema Terra-Sol (validação)
- criar_sistema_apophis()        → Asteroide Apophis (2029)
- criar_sistema_impacto()        → Cenário de impacto hipotético
- criar_sistema_personalizado()  → Sistema com parâmetros customizados
- criar_sistema_aleatorio()      → Condições iniciais aleatórias

📊 FUNÇÕES DE VISUALIZAÇÃO:

- plotar_trajetorias(sistema)           → Órbitas em 2D
- plotar_distancia_temporal(sistema)    → Distância vs tempo
- plotar_conservacao_energia(sistema)   → Validação física
- plotar_resultados_monte_carlo(stats)  → Análise estatística

🎲 SIMULAÇÃO MONTE CARLO:

- simulacao_monte_carlo(n_simulacoes, variacao_posicao, variacao_velocidade)
  → Executa múltiplas simulações com variações aleatórias
  → Retorna estatísticas de risco de impacto

💾 FUNÇÕES DE I/O:

- salvar_configuracao(sistema, arquivo)   → Salva em JSON
- carregar_configuracao(arquivo)          → Carrega de JSON
- salvar_resultado(resultado, arquivo)    → Salva relatório
- exportar_trajetorias(sistema, arquivo)  → Exporta dados

🧪 TESTES DE VALIDAÇÃO:

- teste_conservacao_energia()    → Valida conservação de energia
- teste_terceira_lei_kepler()    → Valida período orbital
- teste_orbita_estavel()         → Valida estabilidade
- executar_todos_testes()        → Executa todos os testes

📖 EXEMPLOS DE USO:

executar_simulacao_interativa()  - Menu interativo completo

Simulação Básica:
```python
sistema = criar_sistema_terra_sol()
resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS)
print(resultado.gerar_relatorio())
plotar_trajetorias(sistema)
```

Apophis:
```python
sistema = criar_sistema_apophis()
resultado = sistema.simular(3 * ANOS_EM_SEGUNDOS)
plotar_distancia_temporal(sistema)
```

Monte Carlo:
```python
stats = simulacao_monte_carlo(n_simulacoes)
plotar_resultados_monte_carlo(stats)
plotar_trajetorias_monte_carlo(stats)
```

⚡ CONSTANTES DISPONÍVEIS:

- G             → Constante gravitacional (6.674e-11)
- UA            → Unidade Astronômica (1.496e11 m)
- M_SOL         → Massa do Sol (1.989e30 kg)
- M_TERRA       → Massa da Terra (5.972e24 kg)
- R_TERRA       → Raio da Terra (6.371e6 m)
- ANOS_EM_SEGUNDOS → Segundos em um ano (31557600)

════════════════════════════════════════════════════════════════════
