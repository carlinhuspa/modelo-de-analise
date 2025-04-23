import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class ModelosEstatisticos:
    """
    Classe com implementações de modelos estatísticos para análise de apostas de futebol.
    Inclui modelos de Poisson, Dixon-Coles e simulação Monte Carlo.
    """
    
    @staticmethod
    def calcular_probabilidades_poisson(lambda_casa, lambda_visitante, max_gols=5):
        """
        Calcula as probabilidades de placares usando distribuição de Poisson.
        
        Parâmetros:
        lambda_casa (float): Média de gols esperados para o time da casa
        lambda_visitante (float): Média de gols esperados para o time visitante
        max_gols (int): Número máximo de gols a considerar
        
        Retorna:
        dict: Dicionário com as probabilidades para cada placar
        """
        probabilidades = {}
        total_prob = 0
        
        for i in range(max_gols + 1):
            for j in range(max_gols + 1):
                prob = poisson.pmf(i, lambda_casa) * poisson.pmf(j, lambda_visitante)
                probabilidades[(i, j)] = prob
                total_prob += prob
        
        # Normalizar probabilidades
        for key in probabilidades:
            probabilidades[key] = probabilidades[key] / total_prob
        
        # Ordenar por probabilidade
        probabilidades_ordenadas = {k: v for k, v in sorted(probabilidades.items(), key=lambda item: item[1], reverse=True)}
        
        return probabilidades_ordenadas
    
    @staticmethod
    def calcular_probabilidades_dixon_coles(lambda_casa, lambda_visitante, rho=-0.1, max_gols=5):
        """
        Calcula as probabilidades de placares usando o modelo Dixon-Coles.
        Este modelo ajusta a distribuição de Poisson para corrigir a dependência entre os gols dos times.
        
        Parâmetros:
        lambda_casa (float): Média de gols esperados para o time da casa
        lambda_visitante (float): Média de gols esperados para o time visitante
        rho (float): Parâmetro de correlação entre os gols dos times
        max_gols (int): Número máximo de gols a considerar
        
        Retorna:
        dict: Dicionário com as probabilidades para cada placar
        """
        def tau(x, y, lambda_x, lambda_y, rho):
            if x == 0 and y == 0:
                return 1 - lambda_x * lambda_y * rho
            elif x == 0 and y == 1:
                return 1 + lambda_x * rho
            elif x == 1 and y == 0:
                return 1 + lambda_y * rho
            elif x == 1 and y == 1:
                return 1 - rho
            else:
                return 1
        
        probabilidades = {}
        total_prob = 0
        
        for i in range(max_gols + 1):
            for j in range(max_gols + 1):
                prob = poisson.pmf(i, lambda_casa) * poisson.pmf(j, lambda_visitante) * tau(i, j, lambda_casa, lambda_visitante, rho)
                probabilidades[(i, j)] = prob
                total_prob += prob
        
        # Normalizar probabilidades
        for key in probabilidades:
            probabilidades[key] = probabilidades[key] / total_prob
        
        # Ordenar por probabilidade
        probabilidades_ordenadas = {k: v for k, v in sorted(probabilidades.items(), key=lambda item: item[1], reverse=True)}
        
        return probabilidades_ordenadas
    
    @staticmethod
    def simular_monte_carlo(lambda_casa, lambda_visitante, n_simulacoes=10000, rho=-0.1, usar_dixon_coles=True):
        """
        Realiza simulação de Monte Carlo para estimar probabilidades de resultados.
        
        Parâmetros:
        lambda_casa (float): Média de gols esperados para o time da casa
        lambda_visitante (float): Média de gols esperados para o time visitante
        n_simulacoes (int): Número de simulações a realizar
        rho (float): Parâmetro de correlação para o modelo Dixon-Coles
        usar_dixon_coles (bool): Se True, usa o modelo Dixon-Coles; se False, usa Poisson simples
        
        Retorna:
        dict: Dicionário com as probabilidades estimadas
        """
        # Obter probabilidades de placares
        if usar_dixon_coles:
            probs_placares = ModelosEstatisticos.calcular_probabilidades_dixon_coles(lambda_casa, lambda_visitante, rho)
        else:
            probs_placares = ModelosEstatisticos.calcular_probabilidades_poisson(lambda_casa, lambda_visitante)
        
        # Preparar para simulação
        placares = list(probs_placares.keys())
        probabilidades = list(probs_placares.values())
        
        # Realizar simulações
        resultados = np.random.choice(len(placares), size=n_simulacoes, p=probabilidades)
        
        # Contar resultados
        vitoria_casa = 0
        empate = 0
        vitoria_visitante = 0
        over_2_5 = 0
        btts = 0
        gol_1_tempo = 0  # Simplificação: assumimos 50% de chance de gol no 1º tempo
        
        for idx in resultados:
            gols_casa, gols_visitante = placares[idx]
            
            # Contar resultados
            if gols_casa > gols_visitante:
                vitoria_casa += 1
            elif gols_casa == gols_visitante:
                empate += 1
            else:
                vitoria_visitante += 1
            
            # Contar over 2.5
            if gols_casa + gols_visitante > 2.5:
                over_2_5 += 1
            
            # Contar BTTS
            if gols_casa > 0 and gols_visitante > 0:
                btts += 1
            
            # Simplificação para gol no 1º tempo
            if np.random.random() < 0.5 and (gols_casa + gols_visitante) > 0:
                gol_1_tempo += 1
        
        # Calcular probabilidades
        return {
            'vitoria_casa': vitoria_casa / n_simulacoes,
            'empate': empate / n_simulacoes,
            'vitoria_visitante': vitoria_visitante / n_simulacoes,
            'over_2_5': over_2_5 / n_simulacoes,
            'btts': btts / n_simulacoes,
            'gol_1_tempo': gol_1_tempo / n_simulacoes
        }
    
    @staticmethod
    def calcular_valor_esperado(probabilidade, odd):
        """
        Calcula o valor esperado (EV) de uma aposta.
        
        Parâmetros:
        probabilidade (float): Probabilidade estimada do evento
        odd (float): Odd oferecida para o evento
        
        Retorna:
        float: Valor esperado da aposta
        """
        return probabilidade * odd - 1
    
    @staticmethod
    def calcular_odds_justas(probabilidade):
        """
        Calcula a odd justa com base na probabilidade estimada.
        
        Parâmetros:
        probabilidade (float): Probabilidade estimada do evento
        
        Retorna:
        float: Odd justa para o evento
        """
        if probabilidade <= 0:
            return float('inf')
        return 1 / probabilidade
    
    @staticmethod
    def calcular_kelly(probabilidade, odd, fracao=1.0):
        """
        Calcula a porcentagem de Kelly para uma aposta.
        
        Parâmetros:
        probabilidade (float): Probabilidade estimada do evento
        odd (float): Odd oferecida para o evento
        fracao (float): Fração de Kelly a utilizar (padrão: 1.0 = Kelly completo)
        
        Retorna:
        float: Porcentagem do bankroll a apostar
        """
        q = 1 - probabilidade
        b = odd - 1  # Odd decimal para odd fracionária
        
        if b * probabilidade > q:
            kelly = (b * probabilidade - q) / b
            return kelly * fracao
        else:
            return 0.0
    
    @staticmethod
    def plotar_matriz_probabilidades(probabilidades, time_casa, time_visitante, max_gols=5):
        """
        Plota uma matriz de calor com as probabilidades para diferentes placares.
        
        Parâmetros:
        probabilidades (dict): Dicionário com as probabilidades para cada placar
        time_casa (str): Nome do time da casa
        time_visitante (str): Nome do time visitante
        max_gols (int): Número máximo de gols a considerar
        
        Retorna:
        matplotlib.figure.Figure: Figura com a matriz de calor
        """
        # Criar matriz de probabilidades
        matriz = np.zeros((max_gols + 1, max_gols + 1))
        for (i, j), prob in probabilidades.items():
            if i <= max_gols and j <= max_gols:
                matriz[i, j] = prob
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, cmap='YlGnBu', fmt='.1%', 
                    xticklabels=range(max_gols + 1), 
                    yticklabels=range(max_gols + 1))
        
        # Configurar rótulos
        ax.set_xlabel(f'Gols {time_visitante}')
        ax.set_ylabel(f'Gols {time_casa}')
        ax.set_title('Probabilidades de Placares')
        
        return fig
    
    @staticmethod
    def ajustar_parametros_dixon_coles(historico_jogos, max_iter=100):
        """
        Ajusta os parâmetros do modelo Dixon-Coles com base no histórico de jogos.
        Implementação simplificada para demonstração.
        
        Parâmetros:
        historico_jogos (list): Lista de tuplas (time_casa, time_visitante, gols_casa, gols_visitante)
        max_iter (int): Número máximo de iterações
        
        Retorna:
        tuple: (ataque_casa, defesa_casa, ataque_visitante, defesa_visitante, rho)
        """
        # Implementação simplificada - em um caso real, usaríamos máxima verossimilhança
        # Esta é apenas uma demonstração do conceito
        
        # Inicializar parâmetros
        ataque = {}
        defesa = {}
        home_advantage = 1.2
        rho = -0.1
        
        # Extrair times únicos
        times = set()
        for time_casa, time_visitante, _, _ in historico_jogos:
            times.add(time_casa)
            times.add(time_visitante)
        
        # Inicializar parâmetros
        for time in times:
            ataque[time] = 1.0
            defesa[time] = 1.0
        
        # Iterações para ajustar parâmetros
        for _ in range(max_iter):
            # Em uma implementação real, ajustaríamos os parâmetros aqui
            pass
        
        # Retornar parâmetros ajustados (simplificado)
        return ataque, defesa, home_advantage, rho
