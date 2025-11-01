# ==============================================================================
# SIMULADOR ORBITAL DE ASTEROIDES
# Projeto de Computação Científica
# 
# Autores: Raul Augusti Pereira, Valentina Spohr, Otávio Augusto Assugeni Guelfi
# ==============================================================================

# ==============================================================================
# PARTE 1: IMPORTS E CONSTANTES
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constantes Físicas
G = 6.67430e-11  # Constante gravitacional (m³/kg/s²)
UA = 1.496e11    # Unidade Astronômica (m)
M_SOL = 1.989e30  # Massa do Sol (kg)
M_TERRA = 5.972e24  # Massa da Terra (kg)
R_TERRA = 6.371e6  # Raio da Terra (m)
M_LUA = 7.342e22  # Massa da Lua (kg)

# Configurações de Simulação
DT_PADRAO = 3600  # Passo de tempo padrão: 1 hora (s)
TOLERANCIA_ENERGIA = 1e-6  # Tolerância para conservação de energia
ANOS_EM_SEGUNDOS = 365.25 * 24 * 3600

print("✓ Bibliotecas e constantes carregadas com sucesso!")


# ==============================================================================
# PARTE 2: CLASSE CORPOCELESTE
# ==============================================================================

@dataclass
class CorpoCeleste:
    """
    Representa um corpo celeste no sistema gravitacional.
    """
    nome: str
    massa: float  # kg
    posicao: np.ndarray  # [x, y, z] em metros
    velocidade: np.ndarray  # [vx, vy, vz] em m/s
    cor: str = 'blue'
    raio_visual: float = 5.0
    
    # Histórico de trajetória
    historico_posicao: List[np.ndarray] = field(default_factory=list)
    historico_velocidade: List[np.ndarray] = field(default_factory=list)
    historico_tempo: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Inicializa arrays como numpy arrays."""
        self.posicao = np.array(self.posicao, dtype=np.float64)
        self.velocidade = np.array(self.velocidade, dtype=np.float64)
        self.aceleracao = np.zeros(3, dtype=np.float64)
    
    def salvar_estado(self, tempo: float):
        """Salva o estado atual no histórico."""
        self.historico_posicao.append(self.posicao.copy())
        self.historico_velocidade.append(self.velocidade.copy())
        self.historico_tempo.append(tempo)
    
    def energia_cinetica(self) -> float:
        """Calcula a energia cinética do corpo."""
        v_quadrado = np.sum(self.velocidade ** 2)
        return 0.5 * self.massa * v_quadrado
    
    def momento_angular(self, origem: np.ndarray = None) -> np.ndarray:
        """Calcula o momento angular em relação a uma origem."""
        if origem is None:
            origem = np.zeros(3)
        r = self.posicao - origem
        return self.massa * np.cross(r, self.velocidade)
    
    def get_trajetoria_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna arrays x e y da trajetória para plotagem."""
        if not self.historico_posicao:
            return np.array([]), np.array([])
        posicoes = np.array(self.historico_posicao)
        return posicoes[:, 0], posicoes[:, 1]

print("✓ Classe CorpoCeleste definida!")


# ==============================================================================
# PARTE 3: CLASSE RESULTADOSIMULACAO
# ==============================================================================

@dataclass
class ResultadoSimulacao:
    """
    Encapsula todos os resultados de uma simulação.
    """
    distancia_minima: float = float('inf')
    tempo_minima: float = 0.0
    posicao_minima_terra: np.ndarray = field(default_factory=lambda: np.zeros(3))
    posicao_minima_asteroide: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocidade_relativa_minima: float = 0.0
    
    houve_colisao: bool = False
    tempo_colisao: float = 0.0
    velocidade_impacto: float = 0.0
    angulo_impacto: float = 0.0
    energia_impacto: float = 0.0
    equivalente_tnt: float = 0.0
    raio_cratera: float = 0.0
    
    energia_inicial: float = 0.0
    energia_final: float = 0.0
    erro_energia_relativo: float = 0.0
    
    momento_angular_inicial: np.ndarray = field(default_factory=lambda: np.zeros(3))
    momento_angular_final: np.ndarray = field(default_factory=lambda: np.zeros(3))
    erro_momento_relativo: float = 0.0
    
    tempo_simulacao: float = 0.0
    numero_passos: int = 0
    
    def gerar_relatorio(self) -> str:
        """Gera um relatório textual dos resultados."""
        relatorio = []
        relatorio.append("=" * 70)
        relatorio.append("RELATÓRIO DA SIMULAÇÃO ORBITAL".center(70))
        relatorio.append("=" * 70)
        relatorio.append("")
        
        # Informações temporais
        relatorio.append("INFORMAÇÕES TEMPORAIS:")
        relatorio.append(f"  Tempo total simulado: {self.tempo_simulacao/ANOS_EM_SEGUNDOS:.2f} anos")
        relatorio.append(f"  Número de passos: {self.numero_passos:,}")
        relatorio.append("")
        
        # Aproximação mínima
        relatorio.append("APROXIMAÇÃO MÍNIMA:")
        relatorio.append(f"  Distância mínima: {self.distancia_minima/1000:.2f} km")
        relatorio.append(f"  Distância em raios terrestres: {self.distancia_minima/R_TERRA:.2f} R⊕")
        relatorio.append(f"  Tempo da aproximação: {self.tempo_minima/ANOS_EM_SEGUNDOS:.4f} anos")
        relatorio.append(f"  Velocidade relativa: {self.velocidade_relativa_minima/1000:.2f} km/s")
        relatorio.append("")
        
        # Colisão (se houver)
        if self.houve_colisao:
            # Verificar se foi queda no Sol (flag: angulo_impacto == -1)
            if self.angulo_impacto == -1:
                relatorio.append("🔥 ASTEROIDE CAIU NO SOL!")
                relatorio.append(f"  Tempo da queda: {self.tempo_colisao/ANOS_EM_SEGUNDOS:.4f} anos")
                relatorio.append(f"  Tempo da queda: {self.tempo_colisao/(24*3600):.1f} dias")
                relatorio.append(f"  Velocidade no impacto: {self.velocidade_impacto/1000:.2f} km/s")
                relatorio.append(f"  Energia liberada: {self.energia_impacto:.2e} J")
            else:
                # Colisão com a Terra
                relatorio.append("⚠️  COLISÃO COM A TERRA DETECTADA!")
                relatorio.append(f"  Tempo de impacto: {self.tempo_colisao/ANOS_EM_SEGUNDOS:.4f} anos")
                relatorio.append(f"  Tempo de impacto: {self.tempo_colisao/(24*3600):.1f} dias")
                relatorio.append(f"  Velocidade de impacto: {self.velocidade_impacto/1000:.2f} km/s")
                relatorio.append(f"  Ângulo de impacto: {self.angulo_impacto:.2f}°")
                relatorio.append(f"  Energia de impacto: {self.energia_impacto:.2e} J")
                relatorio.append(f"  Equivalente em TNT: {self.equivalente_tnt:.2e} megatons")
                relatorio.append(f"  Raio estimado da cratera: {self.raio_cratera/1000:.2f} km")
        else:
            relatorio.append("✓ Nenhuma colisão detectada")
        
        # Conservação de energia
        relatorio.append("VALIDAÇÃO FÍSICA:")
        relatorio.append(f"  Energia inicial: {self.energia_inicial:.6e} J")
        relatorio.append(f"  Energia final: {self.energia_final:.6e} J")
        relatorio.append(f"  Erro relativo de energia: {self.erro_energia_relativo:.2e}")
        if abs(self.erro_energia_relativo) < TOLERANCIA_ENERGIA:
            relatorio.append("  ✓ Energia conservada dentro da tolerância")
        else:
            relatorio.append("  ⚠️  Aviso: Violação na conservação de energia")

        relatorio.append("OBSERVAÇÃO:")
        relatorio.append("🔵 Posição inicial: círculo")
        relatorio.append("⬛ Posição final: quadrado")
        relatorio.append("")
        
        relatorio.append("=" * 70)
        
        return "\n".join(relatorio)

print("✓ Classe ResultadoSimulacao definida!")


# ==============================================================================
# PARTE 4: CLASSE SISTEMAGRAVITACIONAL
# ==============================================================================

class SistemaGravitacional:
    """
    Gerencia o sistema de múltiplos corpos e a evolução temporal.
    """
    
    def __init__(self, dt: float = DT_PADRAO):
        self.corpos: List[CorpoCeleste] = []
        self.dt = dt
        self.tempo_atual = 0.0
        self.resultado = ResultadoSimulacao()
        
    def adicionar_corpo(self, corpo: CorpoCeleste):
        """Adiciona um corpo ao sistema."""
        self.corpos.append(corpo)
    
    def calcular_forca_gravitacional(self, corpo1: CorpoCeleste, 
                                     corpo2: CorpoCeleste) -> np.ndarray:
        """
        Calcula a força gravitacional que corpo2 exerce sobre corpo1.
        F = G * m1 * m2 / r² * r_hat
        """
        r_vec = corpo2.posicao - corpo1.posicao
        r_mag = np.linalg.norm(r_vec)
        
        if r_mag < 1e3:  # Evitar divisão por zero (distância mínima: 1 km)
            return np.zeros(3)
        
        r_hat = r_vec / r_mag
        f_mag = G * corpo1.massa * corpo2.massa / (r_mag ** 2)
        
        return f_mag * r_hat / corpo1.massa  # Retorna aceleração
    
    def calcular_aceleracoes(self):
        """Calcula as acelerações de todos os corpos."""
        # Zerar acelerações
        for corpo in self.corpos:
            corpo.aceleracao = np.zeros(3)
        
        # Calcular forças entre todos os pares
        n = len(self.corpos)
        for i in range(n):
            for j in range(i + 1, n):
                # Força que j exerce sobre i
                a_i = self.calcular_forca_gravitacional(self.corpos[i], self.corpos[j])
                # Força que i exerce sobre j (terceira lei de Newton)
                a_j = -a_i * self.corpos[i].massa / self.corpos[j].massa
                
                self.corpos[i].aceleracao += a_i
                self.corpos[j].aceleracao += a_j
    
    def get_estado(self) -> np.ndarray:
        """Retorna o estado atual do sistema como vetor."""
        estado = []
        for corpo in self.corpos:
            estado.extend(corpo.posicao)
            estado.extend(corpo.velocidade)
        return np.array(estado, dtype=np.float64)
    
    def set_estado(self, estado: np.ndarray):
        """Define o estado do sistema a partir de um vetor."""
        idx = 0
        for corpo in self.corpos:
            corpo.posicao = estado[idx:idx+3].copy()
            corpo.velocidade = estado[idx+3:idx+6].copy()
            idx += 6
    
    def derivada(self, estado: np.ndarray) -> np.ndarray:
        """
        Calcula a derivada temporal do estado.
        Para cada corpo: d/dt[posição, velocidade] = [velocidade, aceleração]
        """
        # Salvar estado atual
        estado_original = self.get_estado()
        
        # Aplicar estado fornecido
        self.set_estado(estado)
        
        # Calcular acelerações
        self.calcular_aceleracoes()
        
        # Construir vetor de derivadas
        derivadas = []
        for corpo in self.corpos:
            derivadas.extend(corpo.velocidade)  # d(posição)/dt = velocidade
            derivadas.extend(corpo.aceleracao)  # d(velocidade)/dt = aceleração
        
        # Restaurar estado original
        self.set_estado(estado_original)
        
        return np.array(derivadas, dtype=np.float64)
    
    def integrador_rk4(self) -> np.ndarray:
        """
        Implementa o método de Runge-Kutta de 4ª ordem.
        """
        y = self.get_estado()
        
        k1 = self.derivada(y)
        k2 = self.derivada(y + 0.5 * self.dt * k1)
        k3 = self.derivada(y + 0.5 * self.dt * k2)
        k4 = self.derivada(y + self.dt * k3)
        
        y_novo = y + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return y_novo
    
    def energia_cinetica_total(self) -> float:
        """Calcula a energia cinética total do sistema."""
        return sum(corpo.energia_cinetica() for corpo in self.corpos)
    
    def energia_potencial_total(self) -> float:
        """Calcula a energia potencial gravitacional total do sistema."""
        ep_total = 0.0
        n = len(self.corpos)
        
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(self.corpos[j].posicao - self.corpos[i].posicao)
                if r > 1e3:  # Evitar divisão por zero
                    ep = -G * self.corpos[i].massa * self.corpos[j].massa / r
                    ep_total += ep
        
        return ep_total
    
    def energia_total(self) -> float:
        """Calcula a energia total do sistema."""
        return self.energia_cinetica_total() + self.energia_potencial_total()
    
    def momento_angular_total(self) -> np.ndarray:
        """Calcula o momento angular total do sistema em relação ao centro de massa."""
        centro_massa = self.centro_de_massa()
        l_total = np.zeros(3)
        
        for corpo in self.corpos:
            l_total += corpo.momento_angular(origem=centro_massa)
        
        return l_total
    
    def centro_de_massa(self) -> np.ndarray:
        """Calcula o centro de massa do sistema."""
        massa_total = sum(corpo.massa for corpo in self.corpos)
        cm = np.zeros(3)
        
        for corpo in self.corpos:
            cm += corpo.massa * corpo.posicao
        
        return cm / massa_total
    
    def detectar_aproximacao(self, corpo1_nome: str = "Terra", 
                        corpo2_nome: str = "Asteroide"):
        """Detecta e registra a aproximação mínima entre dois corpos."""
        corpo1 = next((c for c in self.corpos if c.nome == corpo1_nome), None)
        corpo2 = next((c for c in self.corpos if c.nome == corpo2_nome), None)
        sol = next((c for c in self.corpos if c.nome == "Sol"), None)
        
        if corpo1 is None or corpo2 is None:
            return
        
        # calcular distância
        distancia = np.linalg.norm(corpo2.posicao - corpo1.posicao)
        
        # 1. VERIFICAR QUEDA NO SOL
        if sol is not None and not self.resultado.houve_colisao:
            distancia_sol = np.linalg.norm(corpo2.posicao - sol.posicao)
            RAIO_SOL = 6.96e8  # metros (696,000 km)
            
            if distancia_sol < RAIO_SOL * 2:  # Margem de segurança
                self.resultado.houve_colisao = True
                self.resultado.tempo_colisao = self.tempo_atual
                self.resultado.distancia_minima = distancia_sol
                self.resultado.velocidade_impacto = np.linalg.norm(corpo2.velocidade)
                self.resultado.energia_impacto = 0.5 * corpo2.massa * self.resultado.velocidade_impacto**2
                self.resultado.equivalente_tnt = self.resultado.energia_impacto / 4.184e15
                self.resultado.angulo_impacto = -1  # Flag para identificar queda no Sol
                return  # Parar aqui, não checar Terra
        
        # 2. VERIFICAR COLISÃO COM A TERRA
        if not self.resultado.houve_colisao:
            MARGEM_COLISAO = R_TERRA * 2  # para considerar colisão ao entrar na atmosfera e efeitos numéricos
            
            if distancia < MARGEM_COLISAO:
                self.resultado.houve_colisao = True
                self.resultado.tempo_colisao = self.tempo_atual
                v_relativa = corpo2.velocidade - corpo1.velocidade
                self.calcular_parametros_impacto(corpo1, corpo2, v_relativa)
        
        # 3. atualizar distância mínima
        if distancia < self.resultado.distancia_minima:
            self.resultado.distancia_minima = distancia
            self.resultado.tempo_minima = self.tempo_atual
            self.resultado.posicao_minima_terra = corpo1.posicao.copy()
            self.resultado.posicao_minima_asteroide = corpo2.posicao.copy()
            
            v_relativa = corpo2.velocidade - corpo1.velocidade
            self.resultado.velocidade_relativa_minima = np.linalg.norm(v_relativa)
            
    def calcular_parametros_impacto(self, terra: CorpoCeleste, 
                                   asteroide: CorpoCeleste, 
                                   v_relativa: np.ndarray):
        """Calcula parâmetros físicos do impacto."""
        v_impacto = np.linalg.norm(v_relativa)
        self.resultado.velocidade_impacto = v_impacto
        
        # Energia cinética de impacto
        self.resultado.energia_impacto = 0.5 * asteroide.massa * v_impacto**2
        
        # Equivalente em TNT (1 megaton = 4.184e15 J)
        self.resultado.equivalente_tnt = self.resultado.energia_impacto / 4.184e15
        
        # Ângulo de impacto
        r_vec = asteroide.posicao - terra.posicao
        if np.linalg.norm(r_vec) > 0:
            cos_angulo = np.dot(v_relativa, r_vec) / (v_impacto * np.linalg.norm(r_vec))
            self.resultado.angulo_impacto = np.degrees(np.arccos(np.clip(cos_angulo, -1, 1)))
        
        # Estimativa do raio da cratera (fórmula simplificada)
        densidade_asteroide = 3000  # kg/m³ (aproximação)
        g = 9.81  # m/s²
        
        self.resultado.raio_cratera = (1.8 * 
                                       (densidade_asteroide**(-1/3)) * 
                                       (asteroide.massa**(1/3)) * 
                                       (v_impacto**(2/3)) * 
                                       (g**(-1/3)))
    
    def simular(self, tempo_total: float, progresso: bool = True):
        """
        Executa a simulação por um tempo total especificado.
        """
        # Inicializar resultado
        self.resultado.energia_inicial = self.energia_total()
        self.resultado.momento_angular_inicial = self.momento_angular_total()
        
        # Salvar estados iniciais
        for corpo in self.corpos:
            corpo.salvar_estado(self.tempo_atual)
        
        # Número de passos
        n_passos = int(tempo_total / self.dt)
        self.resultado.numero_passos = n_passos
        self.resultado.tempo_simulacao = tempo_total
        
        # Loop principal
        for passo in range(n_passos):
            # Integrar um passo
            novo_estado = self.integrador_rk4()
            self.set_estado(novo_estado)
            self.tempo_atual += self.dt
            
            # Salvar histórico
            for corpo in self.corpos:
                corpo.salvar_estado(self.tempo_atual)
            
            # Detectar eventos (colisões)
            self.detectar_aproximacao()
            
            # Parar e printar se houve colisão ou queda no Sol
            if self.resultado.houve_colisao:
                if self.resultado.angulo_impacto == -1:
                    # Queda no Sol
                    if progresso:
                        print(f"\n🔥 ASTEROIDE CAIU NO SOL no passo {passo}!")
                        print(f"   Tempo: {self.tempo_atual/(24*3600):.2f} dias ({self.tempo_atual/ANOS_EM_SEGUNDOS:.4f} anos)")
                else:
                    # Colisão com a Terra
                    if progresso:
                        print(f"\n💥 COLISÃO COM A TERRA DETECTADA no passo {passo}!")
                        print(f"   Tempo: {self.tempo_atual/(24*3600):.2f} dias ({self.tempo_atual/ANOS_EM_SEGUNDOS:.4f} anos)")
                break
            
            # Verificar conservação de energia (a cada 100 passos)
            if passo % 100 == 0:
                energia_atual = self.energia_total()
                erro_rel = abs(energia_atual - self.resultado.energia_inicial) / abs(self.resultado.energia_inicial)
                
                if erro_rel > TOLERANCIA_ENERGIA and progresso:
                    print(f"⚠️  Aviso: Erro de conservação = {erro_rel:.2e} no passo {passo}")
            
            # Mostrar progresso
            if progresso and passo % (n_passos // 20) == 0:
                percentual = 100 * passo / n_passos
                print(f"Progresso: {percentual:.1f}% ({passo}/{n_passos} passos)")
        
        # Cálculos finais
        self.resultado.energia_final = self.energia_total()
        self.resultado.erro_energia_relativo = ((self.resultado.energia_final - 
                                                 self.resultado.energia_inicial) / 
                                                abs(self.resultado.energia_inicial))
        
        self.resultado.momento_angular_final = self.momento_angular_total()
        self.resultado.tempo_simulacao = self.tempo_atual  # Tempo real simulado
        
        if progresso:
            print("\n✓ Simulação concluída!")
        
        return self.resultado
        print("✓ Classe SistemaGravitacional completa!")


# ==============================================================================
# CÉLULA 9: FUNÇÕES DE CONFIGURAÇÃO
# ==============================================================================

def criar_sistema_terra_sol() -> SistemaGravitacional:
    """Cria um sistema simples Terra-Sol para validação."""
    sistema = SistemaGravitacional(dt=3600)
    
    # Sol no centro
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra em órbita circular
    v_orbital = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital, 0],
        cor='blue',
        raio_visual=10
    )
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    
    return sistema

def criar_sistema_apophis() -> SistemaGravitacional:
    """
    Cria sistema com asteroide Apophis (99942).
    Aproximação em 2029-04-13.
    """
    sistema = SistemaGravitacional(dt=3600)
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0],
        cor='blue',
        raio_visual=10
    )
    
    # Apophis (aproximação próxima)
    a_apophis = 0.92 * UA
    e_apophis = 0.19
    
    r_perihelio = a_apophis * (1 - e_apophis)
    v_perihelio = np.sqrt(G * M_SOL * (2/r_perihelio - 1/a_apophis))
    
    apophis = CorpoCeleste(
        nome="Asteroide",
        massa=6.1e10,  # ~61 milhões de toneladas
        posicao=[r_perihelio * 0.95, r_perihelio * 0.1, 0],
        velocidade=[-v_perihelio * 0.15, v_perihelio * 0.98, 0],
        cor='red',
        raio_visual=5
    )
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    sistema.adicionar_corpo(apophis)
    
    return sistema

def criar_sistema_impacto() -> SistemaGravitacional:
    """
    Cenário de impacto com a Terra garantido.
    """
    sistema = SistemaGravitacional(dt=900)  # 15 minutos
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra em órbita
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0],  # Positivo = anti-horário
        cor='blue',
        raio_visual=10
    )
    
    # ASTEROIDE: Órbita horária, em direção à Terra!
    
    # Distância orbital similar
    r_asteroide = 1.01992 * UA
    
    # Velocidade orbital para essa distância
    v_orbital_asteroide = np.sqrt(G * M_SOL / r_asteroide)
    
    # Posicionar o asteroide
    angulo_asteroide = np.radians(120)  # 120 graus à frente da Terra
    
    pos_ast_x = r_asteroide * np.cos(angulo_asteroide)
    pos_ast_y = r_asteroide * np.sin(angulo_asteroide)
    
    # Velocidade
    vel_ast_x = v_orbital_asteroide * np.sin(angulo_asteroide) 
    vel_ast_y = -v_orbital_asteroide * np.cos(angulo_asteroide)
    
    # Pequeno ajuste para garantir interceptação exata
    fator_ajuste = 0.98  # 2% mais lento
    vel_ast_x *= fator_ajuste
    vel_ast_y *= fator_ajuste
    
    asteroide = CorpoCeleste(
        nome="Asteroide",
        massa=5e8,  # 500 mil toneladas
        posicao=[pos_ast_x, pos_ast_y, 0],
        velocidade=[vel_ast_x, vel_ast_y, 0],
        cor='red',
        raio_visual=8
    )
    
    # Calcular velocidade relativa de colisão
    v_relativa_estimada = v_orbital_terra + v_orbital_asteroide * fator_ajuste
    
    
    print(f"\n🎯 CENÁRIO DE IMPACTO")
    print(f"   Velocidade Terra: {v_orbital_terra/1000:.2f} km/s")
    print(f"   Velocidade Asteroide: {np.sqrt(vel_ast_x**2 + vel_ast_y**2)/1000:.2f} km/s")
    print(f"   Velocidade relativa de impacto: ~{v_relativa_estimada/1000:.2f} km/s")
    print(f"   Massa: {asteroide.massa:.2e} kg")
    print(f"   dt: {sistema.dt}s ({sistema.dt/60:.0f} min)")
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    sistema.adicionar_corpo(asteroide)
    
    return sistema
    
def criar_sistema_personalizado(massa_asteroide: float,
                                posicao_asteroide: List[float],
                                velocidade_asteroide: List[float]) -> SistemaGravitacional:
    """Cria sistema com parâmetros personalizados."""
    sistema = SistemaGravitacional(dt=3600)
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0],
        cor='blue',
        raio_visual=10
    )
    
    # Asteroide personalizado
    asteroide = CorpoCeleste(
        nome="Asteroide",
        massa=massa_asteroide,
        posicao=posicao_asteroide,
        velocidade=velocidade_asteroide,
        cor='red',
        raio_visual=7
    )
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    sistema.adicionar_corpo(asteroide)
    
    return sistema

def criar_sistema_aleatorio(seed: Optional[int] = None) -> SistemaGravitacional:
    """Cria sistema com condições iniciais aleatórias."""
    if seed is not None:
        np.random.seed(seed)
    
    sistema = SistemaGravitacional(dt=3600)
    
    # Sol
    sol = CorpoCeleste(
        nome="Sol",
        massa=M_SOL,
        posicao=[0, 0, 0],
        velocidade=[0, 0, 0],
        cor='yellow',
        raio_visual=20
    )
    
    # Terra
    v_orbital_terra = np.sqrt(G * M_SOL / UA)
    terra = CorpoCeleste(
        nome="Terra",
        massa=M_TERRA,
        posicao=[UA, 0, 0],
        velocidade=[0, v_orbital_terra, 0],
        cor='blue',
        raio_visual=10
    )
    
    # Asteroide aleatório
    a = np.random.uniform(0.8, 1.5) * UA
    e = np.random.uniform(0.1, 0.4)
    theta = np.random.uniform(0, 2 * np.pi)
    
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    v = np.sqrt(G * M_SOL * (2/r - 1/a))
    
    pos_x = r * np.cos(theta)
    pos_y = r * np.sin(theta)
    vel_x = -v * np.sin(theta)
    vel_y = v * np.cos(theta)
    
    massa = 10 ** np.random.uniform(9, 12)
    
    asteroide = CorpoCeleste(
        nome="Asteroide",
        massa=massa,
        posicao=[pos_x, pos_y, 0],
        velocidade=[vel_x, vel_y, 0],
        cor='red',
        raio_visual=6
    )
    
    sistema.adicionar_corpo(sol)
    sistema.adicionar_corpo(terra)
    sistema.adicionar_corpo(asteroide)
    
    return sistema

print("✓ Funções de configuração criadas!")


# ==============================================================================
# PARTE 5: FUNÇÕES DE VISUALIZAÇÃO
# ==============================================================================

def plotar_trajetorias(sistema: SistemaGravitacional, 
                       titulo: str = "Trajetórias Orbitais",
                       figsize: Tuple[int, int] = (12, 10)):
    """Plota as trajetórias dos corpos em 2D."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotar cada corpo
    for corpo in sistema.corpos:
        x, y = corpo.get_trajetoria_2d()
        if len(x) > 0:
            # Plotar trajetória
            ax.plot(x/UA, y/UA, '-', label=corpo.nome, 
                   color=corpo.cor, linewidth=1.5, alpha=0.7)
            
            # Marcar posição inicial (círculo)
            ax.plot(x[0]/UA, y[0]/UA, 'o', color=corpo.cor, 
                   markersize=corpo.raio_visual, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=1.5)
            
            # Marcar posição final (quadrado)
            ax.plot(x[-1]/UA, y[-1]/UA, 's', color=corpo.cor, 
                   markersize=corpo.raio_visual*0.8, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=1.5)
            
            # Se for Terra ou Asteroide e tiver poucos pontos, destacar mais
            if corpo.nome in ["Terra", "Asteroide"] and len(x) < 100:
                # Plotar com marcadores ao longo da trajetória
                ax.plot(x/UA, y/UA, 'o', color=corpo.cor, 
                       markersize=3, alpha=0.5)
    
    # Marcar ponto de aproximação mínima
    if sistema.resultado.distancia_minima < float('inf'):
        pos_t = sistema.resultado.posicao_minima_terra
        pos_a = sistema.resultado.posicao_minima_asteroide
        
        # Linha tracejada conectando as posições
        ax.plot([pos_t[0]/UA, pos_a[0]/UA], 
               [pos_t[1]/UA, pos_a[1]/UA], 
               'k--', linewidth=2, alpha=0.5, label='Aproximação mínima')
        
        # Estrela no ponto mais próximo do asteroide
        ax.plot(pos_a[0]/UA, pos_a[1]/UA, 'r*', 
               markersize=15, label='Ponto mais próximo',
               markeredgecolor='yellow', markeredgewidth=1)
        
        # Se houve colisão, destacar
        if sistema.resultado.houve_colisao:
            # Círculo vermelho no ponto de colisão
            if sistema.resultado.angulo_impacto != -1:  # Não é queda no Sol
                ax.plot(pos_a[0]/UA, pos_a[1]/UA, 'ro', 
                       markersize=20, alpha=0.3)
                ax.plot(pos_t[0]/UA, pos_t[1]/UA, 'ro', 
                       markersize=20, alpha=0.3)
    
    # Configurar gráfico
    ax.set_xlabel('x (UA)', fontsize=12)
    ax.set_ylabel('y (UA)', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    # Adicionar informações da simulação no título
    if sistema.resultado.houve_colisao:
        if sistema.resultado.angulo_impacto == -1:
            titulo_extra = f"\n Queda no Sol em {sistema.resultado.tempo_colisao/(24*3600):.1f} dias"
        else:
            titulo_extra = f"\n Colisão em {sistema.resultado.tempo_colisao/(24*3600):.1f} dias"
        ax.text(0.5, 1.08, titulo_extra, transform=ax.transAxes,
               ha='center', fontsize=11, color='red', fontweight='bold')
    
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    # Ajustar limites para garantir que tudo é visível
    todas_x = []
    todas_y = []
    for corpo in sistema.corpos:
        x, y = corpo.get_trajetoria_2d()
        if len(x) > 0:
            todas_x.extend(x/UA)
            todas_y.extend(y/UA)
    
    if todas_x and todas_y:
        margem = 0.1
        x_min, x_max = min(todas_x), max(todas_x)
        y_min, y_max = min(todas_y), max(todas_y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        ax.set_xlim(x_min - margem * x_range, x_max + margem * x_range)
        ax.set_ylim(y_min - margem * y_range, y_max + margem * y_range)
    
    plt.tight_layout()
    plt.show()

def plotar_distancia_temporal(sistema: SistemaGravitacional,
                              corpo1_nome: str = "Terra",
                              corpo2_nome: str = "Asteroide"):
    """Plota a evolução da distância entre dois corpos."""
    corpo1 = next((c for c in sistema.corpos if c.nome == corpo1_nome), None)
    corpo2 = next((c for c in sistema.corpos if c.nome == corpo2_nome), None)
    
    if corpo1 is None or corpo2 is None:
        print("Corpos não encontrados!")
        return
    
    # Calcular distâncias
    n_pontos = min(len(corpo1.historico_posicao), len(corpo2.historico_posicao))
    tempos = np.array(corpo1.historico_tempo[:n_pontos]) / ANOS_EM_SEGUNDOS
    distancias = []
    
    for i in range(n_pontos):
        d = np.linalg.norm(corpo2.historico_posicao[i] - corpo1.historico_posicao[i])
        distancias.append(d)
    
    distancias = np.array(distancias)
    
    # Plotar
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tempos, distancias/1e6, 'b-', linewidth=2)
    
    # Marcar distância mínima
    idx_min = np.argmin(distancias)
    ax.plot(tempos[idx_min], distancias[idx_min]/1e6, 'r*', 
           markersize=15, label=f'Mínima: {distancias[idx_min]/1e6:.2f} × 10³ km')
    
    # Linha de referência do raio da Terra
    ax.axhline(R_TERRA/1e6, color='green', linestyle='--', 
              linewidth=2, alpha=0.7, label='Raio da Terra')
    
    ax.set_xlabel('Tempo (anos)', fontsize=12)
    ax.set_ylabel('Distância (× 10³ km)', fontsize=12)
    ax.set_title(f'Distância {corpo1_nome}-{corpo2_nome} ao longo do tempo', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plotar_conservacao_energia(sistema: SistemaGravitacional):
    """Plota a evolução da energia total do sistema."""
    # Recalcular energia em cada ponto
    corpo_referencia = sistema.corpos[0]
    n_pontos = len(corpo_referencia.historico_tempo)
    tempos = np.array(corpo_referencia.historico_tempo) / ANOS_EM_SEGUNDOS
    energias = []
    
    # Salvar estado atual
    estado_original = sistema.get_estado()
    tempo_original = sistema.tempo_atual
    
    # Calcular energia em cada ponto do histórico
    for i in range(0, n_pontos, max(1, n_pontos//1000)):  # Amostragem
        # Restaurar estado histórico
        for j, corpo in enumerate(sistema.corpos):
            if i < len(corpo.historico_posicao):
                corpo.posicao = corpo.historico_posicao[i].copy()
                corpo.velocidade = corpo.historico_velocidade[i].copy()
        
        energias.append(sistema.energia_total())
    
    # Restaurar estado original
    sistema.set_estado(estado_original)
    sistema.tempo_atual = tempo_original
    
    energias = np.array(energias)
    tempos_amostrados = tempos[::max(1, n_pontos//1000)][:len(energias)]
    
    # Calcular erro relativo
    e0 = energias[0]
    erros_relativos = (energias - e0) / abs(e0)
    
    # Plotar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Energia total
    ax1.plot(tempos_amostrados, energias, 'b-', linewidth=2)
    ax1.set_xlabel('Tempo (anos)', fontsize=12)
    ax1.set_ylabel('Energia Total (J)', fontsize=12)
    ax1.set_title('Conservação de Energia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Erro relativo
    ax2.plot(tempos_amostrados, erros_relativos, 'r-', linewidth=2)
    ax2.axhline(TOLERANCIA_ENERGIA, color='green', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Tolerância: {TOLERANCIA_ENERGIA:.1e}')
    ax2.axhline(-TOLERANCIA_ENERGIA, color='green', linestyle='--', 
               linewidth=2, alpha=0.7)
    ax2.set_xlabel('Tempo (anos)', fontsize=12)
    ax2.set_ylabel('Erro Relativo (ΔE/E₀)', fontsize=12)
    ax2.set_title('Erro na Conservação de Energia', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


print("✓ Funções de visualização criadas!")


# ==============================================================================
# CÉLULA 12: SIMULAÇÃO MONTE CARLO
# ==============================================================================

def simulacao_monte_carlo(n_simulacoes: int = 100,
                         variacao_posicao: float = 0.01,
                         variacao_velocidade: float = 0.01,
                         tempo_total: float = 2 * ANOS_EM_SEGUNDOS,
                         seed: Optional[int] = None) -> Dict:
    """
    Executa múltiplas simulações com variações nas condições iniciais.
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Iniciando simulação Monte Carlo com {n_simulacoes} cenários...")
    print("=" * 70)
    
    # Listas para armazenar resultados
    distancias_minimas = []
    tempos_minimos = []
    colisoes = 0
    energias_impacto = []
    
    # Armazenar trajetórias de todos os asteroides
    trajetorias_asteroides = []
    sistema_referencia = None  # Guardar o primeiro sistema para Terra e Sol
    
    # Sistema base
    sistema_base = criar_sistema_apophis()
    asteroide_base = next(c for c in sistema_base.corpos if c.nome == "Asteroide")
    
    pos_base = asteroide_base.posicao.copy()
    vel_base = asteroide_base.velocidade.copy()
    massa_base = asteroide_base.massa
    
    for i in range(n_simulacoes):
        # Variações aleatórias
        delta_pos = np.random.normal(0, variacao_posicao, 3)
        delta_vel = np.random.normal(0, variacao_velocidade, 3)
        
        nova_pos = pos_base * (1 + delta_pos)
        nova_vel = vel_base * (1 + delta_vel)
        
        # Criar sistema com variação
        sistema = criar_sistema_personalizado(massa_base, nova_pos.tolist(), nova_vel.tolist())
        
        # Simular
        resultado = sistema.simular(tempo_total, progresso=False)
        
        # Coletar dados
        distancias_minimas.append(resultado.distancia_minima)
        tempos_minimos.append(resultado.tempo_minima)
        
        if resultado.houve_colisao:
            colisoes += 1
            energias_impacto.append(resultado.energia_impacto)
        
        # Salvar trajetória do asteroide
        asteroide = next(c for c in sistema.corpos if c.nome == "Asteroide")
        trajetorias_asteroides.append({
            'posicoes': asteroide.historico_posicao.copy(),
            'colidiu': resultado.houve_colisao
        })
        
        # Guardar primeiro sistema para referência
        if i == 0:
            sistema_referencia = sistema
        
        # Progresso
        if (i + 1) % (n_simulacoes // 10) == 0:
            print(f"Progresso: {100*(i+1)/n_simulacoes:.0f}% ({i+1}/{n_simulacoes} simulações)")
    
    # Estatísticas
    distancias_minimas = np.array(distancias_minimas)
    tempos_minimos = np.array(tempos_minimos)
    
    estatisticas = {
        'n_simulacoes': n_simulacoes,
        'n_colisoes': colisoes,
        'probabilidade_colisao': colisoes / n_simulacoes,
        'distancia_media': np.mean(distancias_minimas),
        'distancia_std': np.std(distancias_minimas),
        'distancia_min': np.min(distancias_minimas),
        'distancia_max': np.max(distancias_minimas),
        'tempo_medio': np.mean(tempos_minimos),
        'distancias': distancias_minimas,
        'tempos': tempos_minimos,
        'energias_impacto': energias_impacto if energias_impacto else [0],
        # NOVO: Adicionar trajetórias
        'trajetorias_asteroides': trajetorias_asteroides,
        'sistema_referencia': sistema_referencia
    }
    
    print("\n" + "=" * 70)
    print("RESULTADOS DA SIMULAÇÃO MONTE CARLO".center(70))
    print("=" * 70)
    print(f"Total de simulações: {n_simulacoes}")
    print(f"Colisões detectadas: {colisoes}")
    print(f"Probabilidade de impacto: {100*estatisticas['probabilidade_colisao']:.2f}%")
    print(f"\nDistância mínima:")
    print(f"  Média: {estatisticas['distancia_media']/1e6:.2f} × 10³ km")
    print(f"  Desvio padrão: {estatisticas['distancia_std']/1e6:.2f} × 10³ km")
    print(f"  Mínima: {estatisticas['distancia_min']/1e6:.2f} × 10³ km")
    print(f"  Máxima: {estatisticas['distancia_max']/1e6:.2f} × 10³ km")
    print(f"  Em raios terrestres: {estatisticas['distancia_media']/R_TERRA:.2f} ± {estatisticas['distancia_std']/R_TERRA:.2f} R⊕")
    print("=" * 70)
    
    return estatisticas

def plotar_resultados_monte_carlo(estatisticas: Dict):
    """Plota os resultados da simulação Monte Carlo."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    distancias = estatisticas['distancias'] / 1e6  # em 10³ km
    tempos = estatisticas['tempos'] / ANOS_EM_SEGUNDOS
    
    # Histograma de distâncias
    axes[0, 0].hist(distancias, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(R_TERRA/1e6, color='red', linestyle='--', 
                      linewidth=2, label='Raio da Terra')
    axes[0, 0].set_xlabel('Distância Mínima (× 10³ km)', fontsize=11)
    axes[0, 0].set_ylabel('Frequência', fontsize=11)
    axes[0, 0].set_title('Distribuição de Distâncias Mínimas', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograma de tempos
    axes[0, 1].hist(tempos, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Tempo de Aproximação (anos)', fontsize=11)
    axes[0, 1].set_ylabel('Frequência', fontsize=11)
    axes[0, 1].set_title('Distribuição de Tempos de Aproximação', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter: distância vs tempo
    axes[1, 0].scatter(tempos, distancias, alpha=0.6, s=30, c='purple')
    axes[1, 0].axhline(R_TERRA/1e6, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7, label='Raio da Terra')
    axes[1, 0].set_xlabel('Tempo de Aproximação (anos)', fontsize=11)
    axes[1, 0].set_ylabel('Distância Mínima (× 10³ km)', fontsize=11)
    axes[1, 0].set_title('Distância vs Tempo', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Estatísticas resumidas
    axes[1, 1].axis('off')
    stats_text = f"""
    ESTATÍSTICAS RESUMIDAS
    
    Simulações: {estatisticas['n_simulacoes']}
    Colisões: {estatisticas['n_colisoes']}
    Prob. Impacto: {100*estatisticas['probabilidade_colisao']:.2f}%
    
    Distância Mínima:
      Média: {estatisticas['distancia_media']/1e6:.2f} × 10³ km
      Std: {estatisticas['distancia_std']/1e6:.2f} × 10³ km
      Min: {estatisticas['distancia_min']/1e6:.2f} × 10³ km
      Max: {estatisticas['distancia_max']/1e6:.2f} × 10³ km
    
    Em Raios Terrestres:
      {estatisticas['distancia_media']/R_TERRA:.2f} ± {estatisticas['distancia_std']/R_TERRA:.2f} R⊕
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                   verticalalignment='center', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plotar_trajetorias_monte_carlo(estatisticas: Dict):
    """Plota todas as trajetórias dos asteroides da simulação Monte Carlo."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Pegar sistema de referência (primeiro)
    sistema_ref = estatisticas['sistema_referencia']
    
    if sistema_ref is None:
        print("⚠️  Nenhum sistema de referência disponível!")
        return
    
    # Plotar SOL (do primeiro sistema)
    sol = next((c for c in sistema_ref.corpos if c.nome == "Sol"), None)
    if sol:
        ax.plot(0, 0, 'o', color='yellow', markersize=25, 
               markeredgecolor='orange', markeredgewidth=2, label='Sol', zorder=10)
    
    # Plotar TERRA (do primeiro sistema)
    terra = next((c for c in sistema_ref.corpos if c.nome == "Terra"), None)
    if terra:
        x_terra, y_terra = terra.get_trajetoria_2d()
        if len(x_terra) > 0:
            ax.plot(x_terra/UA, y_terra/UA, '-', color='blue', 
                   linewidth=2, alpha=0.8, label='Terra', zorder=5)
            # Marcar posição inicial e final
            ax.plot(x_terra[0]/UA, y_terra[0]/UA, 'o', color='blue', 
                   markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=6)
            ax.plot(x_terra[-1]/UA, y_terra[-1]/UA, 's', color='blue', 
                   markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=6)
    
    # Plotar todos os asteroides
    n_colisoes = estatisticas['n_colisoes']
    n_total = estatisticas['n_simulacoes']
    
    for i, traj_data in enumerate(estatisticas['trajetorias_asteroides']):
        posicoes = traj_data['posicoes']
        colidiu = traj_data['colidiu']
        
        if len(posicoes) > 0:
            posicoes_array = np.array(posicoes)
            x = posicoes_array[:, 0] / UA
            y = posicoes_array[:, 1] / UA
            
            # Cor diferente para colisões
            if colidiu:
                cor = 'red'
                alpha = 0.6
                linewidth = 1.2
                zorder = 4
            else:
                cor = 'black'
                alpha = 0.3
                linewidth = 0.6
                zorder = 1
            
            # Plotar trajetória
            ax.plot(x, y, '-', color=cor, linewidth=linewidth, 
                   alpha=alpha, zorder=zorder)
            
            # Marcar apenas início dos asteroides (para não poluir)
            if i == 0:  # Só no primeiro para aparecer na legenda
                if colidiu:
                    ax.plot(x[0], y[0], 'o', color='red', markersize=4, 
                           alpha=0.7, label=f'Asteroides (colisão: {n_colisoes})', zorder=3)
                else:
                    ax.plot(x[0], y[0], 'o', color='gray', markersize=3, 
                           alpha=0.3, label=f'Asteroides (sem colisão: {n_total-n_colisoes})', zorder=2)
    
    # Adicionar círculo da órbita terrestre como referência
    theta = np.linspace(0, 2*np.pi, 100)
    x_circ = np.cos(theta)
    y_circ = np.sin(theta)
    ax.plot(x_circ, y_circ, ':', color='cyan', alpha=0.4, 
           linewidth=1.5, label='Órbita terrestre (1 UA)', zorder=0)
    
    # Configurações do gráfico
    ax.set_xlabel('x (UA)', fontsize=13)
    ax.set_ylabel('y (UA)', fontsize=13)
    ax.set_title(f'Simulação Monte Carlo - {n_total} Trajetórias\n' + 
                f'Colisões: {n_colisoes} ({100*n_colisoes/n_total:.1f}%)', 
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    
    # Adicionar estatísticas no canto
    stats_text = f"n = {n_total}\nColisões = {n_colisoes}\nProb = {100*estatisticas['probabilidade_colisao']:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

print("✓ Funções de Monte Carlo criadas!")


# ==============================================================================
# PARTE 6: FUNÇÕES DE I/O (JSON)
# ==============================================================================

def salvar_configuracao(sistema: SistemaGravitacional, arquivo: str = 'config.json'):
    """Salva a configuração do sistema em JSON."""
    config = {
        'dt': sistema.dt,
        'corpos': []
    }
    
    for corpo in sistema.corpos:
        config['corpos'].append({
            'nome': corpo.nome,
            'massa': corpo.massa,
            'posicao': corpo.posicao.tolist(),
            'velocidade': corpo.velocidade.tolist(),
            'cor': corpo.cor,
            'raio_visual': corpo.raio_visual
        })
    
    with open(arquivo, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Configuração salva em: {arquivo}")

def carregar_configuracao(arquivo: str = 'config.json') -> SistemaGravitacional:
    """Carrega a configuração do sistema de um JSON."""
    with open(arquivo, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    sistema = SistemaGravitacional(dt=config.get('dt', DT_PADRAO))
    
    for corpo_config in config['corpos']:
        corpo = CorpoCeleste(
            nome=corpo_config['nome'],
            massa=corpo_config['massa'],
            posicao=corpo_config['posicao'],
            velocidade=corpo_config['velocidade'],
            cor=corpo_config.get('cor', 'blue'),
            raio_visual=corpo_config.get('raio_visual', 5.0)
        )
        sistema.adicionar_corpo(corpo)
    
    print(f"✓ Configuração carregada de: {arquivo}")
    return sistema

def salvar_resultado(resultado: ResultadoSimulacao, arquivo: str = 'resultado.txt'):
    """Salva o relatório da simulação em arquivo de texto."""
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(resultado.gerar_relatorio())
    
    print(f"✓ Resultado salvo em: {arquivo}")

def exportar_trajetorias(sistema: SistemaGravitacional, arquivo: str = 'trajetorias.json'):
    """Exporta as trajetórias para JSON."""
    dados = {
        'tempo_simulacao': sistema.tempo_atual,
        'corpos': {}
    }
    
    for corpo in sistema.corpos:
        dados['corpos'][corpo.nome] = {
            'massa': corpo.massa,
            'cor': corpo.cor,
            'tempos': corpo.historico_tempo,
            'posicoes': [p.tolist() for p in corpo.historico_posicao],
            'velocidades': [v.tolist() for v in corpo.historico_velocidade]
        }
    
    with open(arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Trajetórias exportadas para: {arquivo}")

print("✓ Funções de I/O criadas!")


# ==============================================================================
# CÉLULA 14: INTERFACE DE MENU INTERATIVO
# ==============================================================================

def menu_principal():
    """Menu interativo principal."""
    print("\n" + "="*70)
    print("SIMULADOR DE TRAJETÓRIAS DE ASTEROIDES".center(70))
    print("="*70)
    print("\nEscolha uma opção:")
    print("\n1. CASOS PRÉ-CONFIGURADOS")
    print("   a) Órbita da Terra (validação)")
    print("   b) Asteroide Apophis (2029)")
    print("   c) Cenário de impacto hipotético")
    print("\n2. CONFIGURAÇÃO PERSONALIZADA")
    print("   d) Inserir parâmetros manualmente")
    print("   e) Gerar condições aleatórias")
    print("\n3. SIMULAÇÃO MONTE CARLO")
    print("   f) Executar análise estatística")
    print("\n4. CARREGAR CONFIGURAÇÃO")
    print("   g) Carregar de arquivo JSON")
    print("\n0. Sair")
    print("="*70)

def executar_simulacao_interativa():
    """Executa o simulador de forma interativa."""
    while True:
        menu_principal()
        opcao = input("\nDigite sua opção: ").strip().lower()
        
        if opcao == '0':
            print("\nEncerrando simulador. Até logo!")
            break
        
        elif opcao == 'a':
            print("\n>>> Simulando órbita da Terra (validação)...")
            sistema = criar_sistema_terra_sol()
            
            tempo = float(input("Tempo de simulação (anos): ") or "2")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            plotar_trajetorias(sistema, "Validação: Órbita da Terra")
            plotar_conservacao_energia(sistema)
            
            if input("\nSalvar resultado? (s/n): ").lower() == 's':
                salvar_resultado(resultado, 'resultado_terra.txt')
        
        elif opcao == 'b':
            print("\n>>> Simulando Asteroide Apophis (2029)...")
            sistema = criar_sistema_apophis()
            
            tempo = float(input("Tempo de simulação (anos): ") or "3")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            plotar_trajetorias(sistema, "Asteroide Apophis - Aproximação 2029")
            plotar_distancia_temporal(sistema)
            plotar_conservacao_energia(sistema)
            
            if input("\nSalvar resultado? (s/n): ").lower() == 's':
                salvar_resultado(resultado, 'resultado_apophis.txt')
                exportar_trajetorias(sistema, 'trajetorias_apophis.json')
        
        elif opcao == 'c':
            print("\n>>> Simulando cenário de impacto hipotético...")
            sistema = criar_sistema_impacto()
            
            tempo = float(input("Tempo de simulação (anos): ") or "0.5")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            plotar_trajetorias(sistema, "Cenário de Impacto Hipotético")
            plotar_distancia_temporal(sistema)
            
            if input("\nSalvar resultado? (s/n): ").lower() == 's':
                salvar_resultado(resultado, 'resultado_impacto.txt')
        
        elif opcao == 'd':
            print("\n>>> Configuração personalizada")
            print("\nInsira os parâmetros do asteroide:")
            
            massa = float(input("  Massa (kg) [ex: 6.1e10]: ") or "6.1e10")
            
            print("  Posição inicial (UA):")
            px = float(input("    x: ") or "1.0") * UA
            py = float(input("    y: ") or "0") * UA
            pz = float(input("    z: ") or "0") * UA
            
            print("  Velocidade inicial (m/s):")
            vx = float(input("    vx: ") or "0")
            vy = float(input("    vy: ") or "30000")
            vz = float(input("    vz: ") or "0")
            
            sistema = criar_sistema_personalizado(massa, [px, py, pz], [vx, vy, vz])
            
            tempo = float(input("\nTempo de simulação (anos): ") or "2")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            plotar_trajetorias(sistema, "Simulação Personalizada")
            plotar_distancia_temporal(sistema)
        
        elif opcao == 'e':
            print("\n>>> Gerando condições aleatórias...")
            seed = input("Seed (deixe vazio para aleatório): ").strip()
            seed = int(seed) if seed else None
            
            sistema = criar_sistema_aleatorio(seed)
            
            print("\nCondições geradas:")
            asteroide = next(c for c in sistema.corpos if c.nome == "Asteroide")
            print(f"  Massa: {asteroide.massa:.2e} kg")
            print(f"  Posição: ({asteroide.posicao[0]/UA:.3f}, {asteroide.posicao[1]/UA:.3f}) UA")
            print(f"  Velocidade: ({asteroide.velocidade[0]/1000:.2f}, {asteroide.velocidade[1]/1000:.2f}) km/s")
            
            tempo = float(input("\nTempo de simulação (anos): ") or "2")
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            resultado = sistema.simular(tempo_total)
            print("\n" + resultado.gerar_relatorio())
            
            plotar_trajetorias(sistema, "Simulação com Condições Aleatórias")
            plotar_distancia_temporal(sistema)
        
        elif opcao == 'f':
            print("\n>>> Simulação Monte Carlo")
            
            n_sim = int(input("Número de simulações [mínimo 10]: ") or "100")
            var_pos = float(input("Variação em posição (%): ") or "1") / 100
            var_vel = float(input("Variação em velocidade (%)]: ") or "1") / 100
            tempo = float(input("Tempo por simulação (anos): ") or "2")
            
            tempo_total = tempo * ANOS_EM_SEGUNDOS
            
            estatisticas = simulacao_monte_carlo(
                n_simulacoes=n_sim,
                variacao_posicao=var_pos,
                variacao_velocidade=var_vel,
                tempo_total=tempo_total
            )
            
            plotar_resultados_monte_carlo(estatisticas)
            plotar_trajetorias_monte_carlo(estatisticas)
        
        elif opcao == 'g':
            print("\n>>> Carregar configuração de arquivo")
            arquivo = input("Nome do arquivo [padrão: config.json]: ").strip() or "config.json"
            
            try:
                sistema = carregar_configuracao(arquivo)
                
                tempo = float(input("Tempo de simulação (anos): ") or "2")
                tempo_total = tempo * ANOS_EM_SEGUNDOS
                
                resultado = sistema.simular(tempo_total)
                print("\n" + resultado.gerar_relatorio())
                
                plotar_trajetorias(sistema, "Simulação Carregada")
                plotar_distancia_temporal(sistema)
            
            except FileNotFoundError:
                print(f"⚠️  Arquivo '{arquivo}' não encontrado!")
            except json.JSONDecodeError:
                print(f"⚠️  Erro ao ler o arquivo JSON!")
        
        else:
            print("\n⚠️  Opção inválida! Tente novamente.")
        
        input("\nPressione ENTER para continuar...")

print("✓ Interface de menu criada!")


# ==============================================================================
# PARTE 7: TESTES DE VALIDAÇÃO
# ==============================================================================

def teste_conservacao_energia():
    """Testa a conservação de energia em órbita circular."""
    print("\nTESTE 1: Conservação de Energia (Órbita Terra-Sol)")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS, progresso=False)
    
    erro_energia = abs(resultado.erro_energia_relativo)
    
    print(f"Energia inicial: {resultado.energia_inicial:.6e} J")
    print(f"Energia final:   {resultado.energia_final:.6e} J")
    print(f"Erro relativo:   {erro_energia:.6e}")
    print(f"Tolerância:      {TOLERANCIA_ENERGIA:.6e}")
    
    if erro_energia < TOLERANCIA_ENERGIA:
        print("✓ TESTE PASSOU: Energia conservada!")
    else:
        print("✗ TESTE FALHOU: Violação na conservação de energia!")
    
    return erro_energia < TOLERANCIA_ENERGIA

def teste_terceira_lei_kepler():
    """Testa a Terceira Lei de Kepler."""
    print("\nTESTE 2: Terceira Lei de Kepler")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(1 * ANOS_EM_SEGUNDOS, progresso=False)
    
    terra = next(c for c in sistema.corpos if c.nome == "Terra")
    
    # Calcular período orbital
    periodo_simulado = terra.historico_tempo[-1]
    
    # Período teórico pela 3ª Lei de Kepler: T² = (4π²/GM) * a³
    a = UA  # semi-eixo maior
    periodo_teorico = 2 * np.pi * np.sqrt(a**3 / (G * M_SOL))
    
    erro_relativo = abs(periodo_simulado - periodo_teorico) / periodo_teorico
    
    print(f"Período simulado: {periodo_simulado/ANOS_EM_SEGUNDOS:.6f} anos")
    print(f"Período teórico:  {periodo_teorico/ANOS_EM_SEGUNDOS:.6f} anos")
    print(f"Erro relativo:    {erro_relativo:.6e}")
    
    if erro_relativo < 0.01:  # 1% de erro
        print("✓ TESTE PASSOU: Período orbital correto!")
    else:
        print("✗ TESTE FALHOU: Período orbital incorreto!")
    
    return erro_relativo < 0.01

def teste_orbita_estavel():
    """Testa se a órbita da Terra permanece estável."""
    print("\nTESTE 3: Estabilidade Orbital")
    print("="*70)
    
    sistema = criar_sistema_terra_sol()
    resultado = sistema.simular(10 * ANOS_EM_SEGUNDOS, progresso=False)
    
    terra = next(c for c in sistema.corpos if c.nome == "Terra")
    
    # Calcular raio orbital ao longo do tempo
    posicoes = np.array(terra.historico_posicao)
    raios = np.linalg.norm(posicoes, axis=1)
    
    raio_medio = np.mean(raios)
    variacao = np.std(raios) / raio_medio
    
    print(f"Raio médio:       {raio_medio/UA:.6f} UA")
    print(f"Raio esperado:    1.000000 UA")
    print(f"Variação relativa: {variacao:.6e}")
    
    if variacao < 0.001:  # 0.1% de variação
        print("✓ TESTE PASSOU: Órbita estável!")
    else:
        print("✗ TESTE FALHOU: Órbita instável!")
    
    return variacao < 0.001

def executar_todos_testes():
    """Executa todos os testes de validação."""
    print("\n" + "="*70)
    print("EXECUTANDO TESTES DE VALIDAÇÃO".center(70))
    print("="*70)
    
    resultados = []
    
    resultados.append(teste_conservacao_energia())
    resultados.append(teste_terceira_lei_kepler())
    resultados.append(teste_orbita_estavel())
    
    print("\n" + "="*70)
    print("RESUMO DOS TESTES".center(70))
    print("="*70)
    print(f"Total de testes: {len(resultados)}")
    print(f"Testes passados: {sum(resultados)}")
    print(f"Testes falhados: {len(resultados) - sum(resultados)}")
    
    if all(resultados):
        print("\n✓ TODOS OS TESTES PASSARAM!")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM!")
    
    print("="*70)
    
    return all(resultados)

print("✓ Testes de validação criados!")


# ==============================================================================
# PARTE 8: EXEMPLOS DE USO
# ==============================================================================

def exemplo_basico():
    """Exemplo básico de uso do simulador."""
    print("\n" + "="*70)
    print("EXEMPLO 1: Simulação Básica (Órbita da Terra)".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_terra_sol()
    
    # Simular por 2 anos
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS)
    
    # Mostrar relatório
    print("\n" + resultado.gerar_relatorio())
    
    # Visualizar
    plotar_trajetorias(sistema)
    plotar_conservacao_energia(sistema)

def exemplo_apophis():
    """Exemplo com asteroide Apophis."""
    print("\n" + "="*70)
    print("EXEMPLO 2: Asteroide Apophis (2029)".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_apophis()
    
    # Simular por 3 anos
    resultado = sistema.simular(3 * ANOS_EM_SEGUNDOS)
    
    # Mostrar relatório
    print("\n" + resultado.gerar_relatorio())
    
    # Visualizações
    plotar_trajetorias(sistema, "Asteroide Apophis - Aproximação 2029")
    plotar_distancia_temporal(sistema)
    plotar_conservacao_energia(sistema)

def exemplo_impacto():
    """Exemplo de cenário de impacto."""
    print("\n" + "="*70)
    print("EXEMPLO 3: Cenário de Impacto Hipotético".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_impacto()
    
    # Simular por 6 meses
    resultado = sistema.simular(0.5 * ANOS_EM_SEGUNDOS)
    
    # Mostrar relatório
    print("\n" + resultado.gerar_relatorio())
    
    # Visualizar
    plotar_trajetorias(sistema, "Cenário de Impacto")
    plotar_distancia_temporal(sistema)

def exemplo_monte_carlo():
    """Exemplo de simulação Monte Carlo."""
    print("\n" + "="*70)
    print("EXEMPLO 4: Simulação Monte Carlo".center(70))
    print("="*70)
    
    # Executar Monte Carlo
    estatisticas = simulacao_monte_carlo(
        n_simulacoes=15,
        variacao_posicao=0.01,
        variacao_velocidade=0.01,
        tempo_total=1 * ANOS_EM_SEGUNDOS,
        seed=42
    )
    
    # Visualizar resultados
    plotar_resultados_monte_carlo(estatisticas)
    plotar_trajetorias_monte_carlo(estatisticas)

def exemplo_personalizado(massa = 5e11, posicao = [1.2 * UA, 0.1 * UA, 0], velocidade = [0, 28000, 0]):
    """Exemplo com configuração personalizada."""
    print("\n" + "="*70)
    print("EXEMPLO 5: Configuração Personalizada".center(70))
    print("="*70)
    
    # Criar sistema
    sistema = criar_sistema_personalizado(massa, posicao, velocidade)
    
    # Simular
    resultado = sistema.simular(2 * ANOS_EM_SEGUNDOS)
    
    # Resultados
    print("\n" + resultado.gerar_relatorio())
    plotar_trajetorias(sistema, "Asteroide Personalizado")
    plotar_distancia_temporal(sistema)

print("✓ Exemplos de uso criados!")


# ==============================================================================
# PARTE 9: MENU DE EXECUÇÃO RÁPIDA
# ==============================================================================

print("\n" + "="*70)
print("📊 SIMULADOR DE TRAJETÓRIAS DE ASTEROIDES".center(70))
print("="*70)
print("\n✓ Todas as funções carregadas com sucesso!")
print("\nOpções de execução:")
print("  1. executar_simulacao_interativa()  - Menu interativo completo")
print("  2. exemplo_basico()                 - Órbita da Terra (validação)")
print("  3. exemplo_apophis()                - Asteroide Apophis")
print("  4. exemplo_impacto()                - Cenário de impacto")
print("  5. exemplo_monte_carlo()            - Análise estatística")
print("  6. exemplo_personalizado()          - Configuração customizada")
print("  7. executar_todos_testes()          - Validação completa")
print("\nExecute qualquer uma dessas funções!")
print("Exemplo: executar_simulacao_interativa()")
print('Opcional: use o "%matplotlib widget" para tornar os gráficos interativos')
print("="*70)


# ==============================================================================
# PARTE 10: DOCUMENTAÇÃO E AJUDA
# ==============================================================================

def mostrar_ajuda():
    """Mostra documentação completa do simulador."""
    ajuda = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║       SIMULADOR ORBITAL DE ASTEROIDES - GUIA DE USO               ║
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
    """
    print(ajuda)


# ==============================================================================
# FIM DO CÓDIGO - SIMULADOR COMPLETO
# ==============================================================================

print("\n" + "="*70)
print("✓ CÓDIGO COMPLETO CARREGADO!".center(70))
print("="*70)
print("\nPara começar, execute:")
print("  • executar_simulacao_interativa()  (menu interativo)")
print("  • exemplo_apophis()                (exemplo rápido)")
print("  • executar_todos_testes()          (validação)")
print("  • mostrar_ajuda()                  (documentação)")
print("="*70)