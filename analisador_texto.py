import re
import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class AnalisadorTextoEstatistico:
    """
    Classe para analisar textos estatísticos de futebol e extrair informações relevantes.
    Suporta análise de textos do FootyStats e outros formatos estatísticos.
    """
    
    def __init__(self):
        """Inicializa o analisador de texto estatístico."""
        self.dados_extraidos = {}
        self.texto_estatistico = None
        self.texto_cientifico = None
        self.times = {'casa': None, 'visitante': None}
        self.odds = {'casa': None, 'empate': None, 'visitante': None}
        self.xg = {'casa': None, 'visitante': None}
        self.gols_marcados = {'casa': None, 'visitante': None}
        self.gols_sofridos = {'casa': None, 'visitante': None}
        self.probabilidades_poisson = {}
        self.probabilidades_monte_carlo = {}
        self.confrontos_diretos = {}
        self.recomendacoes = []
        
    def carregar_texto(self, texto_estatistico):
        """
        Carrega o texto estatístico para análise.
        
        Parâmetros:
        texto_estatistico (str): Texto com estatísticas de futebol (formato FootyStats ou similar)
        """
        self.texto_estatistico = texto_estatistico
        self._extrair_informacoes_basicas()
        return True
    
    def _extrair_informacoes_basicas(self):
        """Extrai informações básicas do texto estatístico."""
        # Extrair nomes dos times
        padrao_confronto = r'(\w+[\w\s]+)\s+x\s+(\w+[\w\s]+)'
        match_confronto = re.search(padrao_confronto, self.texto_estatistico)
        if match_confronto:
            self.times['casa'] = match_confronto.group(1).strip()
            self.times['visitante'] = match_confronto.group(2).strip()
            
        # Extrair odds
        padrao_odds = r'(\d+\.\d+)\s*[WL]\s*[WL]\s*[WLD]\s*[WL]\s*[WL]\s*[\w\s]+\s*(\d+\.\d+)'
        match_odds = re.search(padrao_odds, self.texto_estatistico)
        if match_odds:
            self.odds['casa'] = float(match_odds.group(1))
            self.odds['visitante'] = float(match_odds.group(2))
            
            # Tentar encontrar odd de empate
            padrao_odd_empate = r'Odd_Empate[:\s]+(\d+[\.,]\d+)'
            match_odd_empate = re.search(padrao_odd_empate, self.texto_estatistico)
            if match_odd_empate:
                self.odds['empate'] = float(match_odd_empate.group(1).replace(',', '.'))
            else:
                # Valor padrão se não encontrar
                self.odds['empate'] = 3.25
        
        # Extrair xG
        padrao_xg = r'xG\s*\n*\s*(\d+\.\d+)\s*\n*\s*(\d+\.\d+)'
        match_xg = re.search(padrao_xg, self.texto_estatistico)
        if match_xg:
            self.xg['casa'] = float(match_xg.group(1))
            self.xg['visitante'] = float(match_xg.group(2))
        
        # Extrair estatísticas de gols marcados
        padrao_gols_casa = r'Marcaram\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)'
        match_gols_casa = re.search(padrao_gols_casa, self.texto_estatistico)
        if match_gols_casa:
            self.gols_marcados['casa'] = float(match_gols_casa.group(2))  # Gols em casa
            self.gols_marcados['visitante'] = float(match_gols_casa.group(3))  # Gols fora
        
        # Extrair estatísticas de gols sofridos
        padrao_gols_sofridos = r'Sofreram\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)'
        match_gols_sofridos = re.search(padrao_gols_sofridos, self.texto_estatistico)
        if match_gols_sofridos:
            self.gols_sofridos['casa'] = float(match_gols_sofridos.group(2))  # Gols sofridos em casa
            self.gols_sofridos['visitante'] = float(match_gols_sofridos.group(3))  # Gols sofridos fora
        
        # Extrair confrontos diretos
        padrao_confrontos = r'(\d+)\s*Jogos\s*(\d+)%\s*(\d+)%\s*(\d+)%\s*(\d+)\s*Vitórias\s*(\d+)\s*Empates\s*\((\d+)%\)\s*(\d+)\s*Vitórias'
        match_confrontos = re.search(padrao_confrontos, self.texto_estatistico)
        if match_confrontos:
            self.confrontos_diretos = {
                'total_jogos': int(match_confrontos.group(1)),
                'vitoria_casa_pct': int(match_confrontos.group(2)),
                'empate_pct': int(match_confrontos.group(3)),
                'vitoria_visitante_pct': int(match_confrontos.group(4)),
                'vitoria_casa': int(match_confrontos.group(5)),
                'empates': int(match_confrontos.group(6)),
                'vitoria_visitante': int(match_confrontos.group(8))
            }
        
        # Extrair estatísticas de over/under
        padrao_over = r'(\d+)%Mais de 2.5'
        match_over = re.search(padrao_over, self.texto_estatistico)
        if match_over:
            self.dados_extraidos['over_2_5_pct'] = int(match_over.group(1))
        
        padrao_btts = r'(\d+)%AM'
        match_btts = re.search(padrao_btts, self.texto_estatistico)
        if match_btts:
            self.dados_extraidos['btts_pct'] = int(match_btts.group(1))
        
        # Extrair média de gols por jogo
        padrao_media_gols = r'(\d+\.\d+)Golos / Jogo'
        match_media_gols = re.search(padrao_media_gols, self.texto_estatistico)
        if match_media_gols:
            self.dados_extraidos['media_gols_jogo'] = float(match_media_gols.group(1))
        
        # Extrair estatísticas de cantos
        padrao_cantos = r'(\d+\.\d+)\s*Cantos / jogo'
        match_cantos = re.search(padrao_cantos, self.texto_estatistico)
        if match_cantos:
            self.dados_extraidos['media_cantos_jogo'] = float(match_cantos.group(1))
    
    def carregar_texto_cientifico(self, texto_cientifico):
        """
        Carrega o texto de análise científica para processamento.
        
        Parâmetros:
        texto_cientifico (str): Texto com análise científica (Poisson, xG, etc.)
        """
        self.texto_cientifico = texto_cientifico
        self._extrair_informacoes_cientificas()
        return True
    
    def _extrair_informacoes_cientificas(self):
        """Extrai informações da análise científica."""
        if not self.texto_cientifico:
            return
        
        # Extrair xG médio
        padrao_xg_casa = r'xG médio:\s*(\d+\.\d+)'
        match_xg_casa = re.search(padrao_xg_casa, self.texto_cientifico)
        if match_xg_casa and not self.xg['casa']:
            self.xg['casa'] = float(match_xg_casa.group(1))
        
        padrao_xg_visitante = r'xG médio:\s*(\d+\.\d+)'
        match_xg_visitante = re.search(padrao_xg_visitante, self.texto_cientifico, re.DOTALL)
        if match_xg_visitante and not self.xg['visitante']:
            # Pegar o segundo match se existir
            matches = re.findall(padrao_xg_visitante, self.texto_cientifico)
            if len(matches) > 1:
                self.xg['visitante'] = float(matches[1])
        
        # Extrair probabilidades de Poisson
        padrao_poisson = r'Simulação de Poisson.*?Resultado\s+Probabilidade(.*?)Placar exato mais provável:'
        match_poisson = re.search(padrao_poisson, self.texto_cientifico, re.DOTALL)
        if match_poisson:
            resultados_texto = match_poisson.group(1)
            padrao_resultado = r'(\d+)x(\d+)\s+(\d+\.\d+)%'
            for match in re.finditer(padrao_resultado, resultados_texto):
                gols_casa = int(match.group(1))
                gols_visitante = int(match.group(2))
                probabilidade = float(match.group(3))
                self.probabilidades_poisson[(gols_casa, gols_visitante)] = probabilidade / 100
        
        # Extrair probabilidades de Monte Carlo
        padrao_monte_carlo = r'Simulação Monte Carlo.*?Vitória.*?:\s*(\d+\.\d+)%\s*Empate:\s*(\d+\.\d+)%\s*Vitória.*?:\s*(\d+\.\d+)%\s*Over 2.5 gols:\s*(\d+\.\d+)%\s*BTTS:\s*(\d+\.\d+)%'
        match_monte_carlo = re.search(padrao_monte_carlo, self.texto_cientifico, re.DOTALL)
        if match_monte_carlo:
            self.probabilidades_monte_carlo = {
                'vitoria_casa': float(match_monte_carlo.group(1)) / 100,
                'empate': float(match_monte_carlo.group(2)) / 100,
                'vitoria_visitante': float(match_monte_carlo.group(3)) / 100,
                'over_2_5': float(match_monte_carlo.group(4)) / 100,
                'btts': float(match_monte_carlo.group(5)) / 100
            }
        
        # Extrair recomendações
        padrao_recomendacoes = r'Recomendações finais:(.*?)$'
        match_recomendacoes = re.search(padrao_recomendacoes, self.texto_cientifico, re.DOTALL | re.MULTILINE)
        if match_recomendacoes:
            recomendacoes_texto = match_recomendacoes.group(1)
            padrao_entrada = r'🔵 Entrada.*?:(.*?)(?:🔵|\Z)'
            for match in re.finditer(padrao_entrada, recomendacoes_texto, re.DOTALL):
                self.recomendacoes.append(match.group(1).strip())
    
    def calcular_probabilidades_poisson(self):
        """
        Calcula as probabilidades de placares usando distribuição de Poisson.
        Usa xG como lambda se disponível, caso contrário usa média de gols.
        """
        # Definir lambdas (média de gols esperados)
        lambda_casa = self.xg['casa'] if self.xg['casa'] else self.gols_marcados['casa']
        lambda_visitante = self.xg['visitante'] if self.xg['visitante'] else self.gols_marcados['visitante']
        
        if not lambda_casa or not lambda_visitante:
            return {}
        
        # Calcular probabilidades para cada placar até 5 gols
        max_gols = 5
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
        
        # Atualizar probabilidades de Poisson
        self.probabilidades_poisson = probabilidades_ordenadas
        return probabilidades_ordenadas
    
    def calcular_probabilidades_resultados(self):
        """
        Calcula as probabilidades de resultados (vitória casa, empate, vitória visitante)
        com base nas probabilidades de placares.
        """
        if not self.probabilidades_poisson and not self.probabilidades_monte_carlo:
            self.calcular_probabilidades_poisson()
        
        # Se temos probabilidades de Monte Carlo, usamos elas
        if self.probabilidades_monte_carlo:
            return {
                'Casa': self.probabilidades_monte_carlo.get('vitoria_casa', 0),
                'Empate': self.probabilidades_monte_carlo.get('empate', 0),
                'Visitante': self.probabilidades_monte_carlo.get('vitoria_visitante', 0)
            }
        
        # Caso contrário, calculamos a partir das probabilidades de Poisson
        prob_casa = 0
        prob_empate = 0
        prob_visitante = 0
        
        for (gols_casa, gols_visitante), prob in self.probabilidades_poisson.items():
            if gols_casa > gols_visitante:
                prob_casa += prob
            elif gols_casa == gols_visitante:
                prob_empate += prob
            else:
                prob_visitante += prob
        
        return {
            'Casa': prob_casa,
            'Empate': prob_empate,
            'Visitante': prob_visitante
        }
    
    def calcular_probabilidades_over_under(self):
        """
        Calcula as probabilidades de over/under com base nas probabilidades de placares.
        """
        if not self.probabilidades_poisson and not self.probabilidades_monte_carlo:
            self.calcular_probabilidades_poisson()
        
        # Se temos probabilidades de Monte Carlo para over 2.5, usamos ela
        if self.probabilidades_monte_carlo and 'over_2_5' in self.probabilidades_monte_carlo:
            return {
                'Over 2.5': self.probabilidades_monte_carlo['over_2_5'],
                'Under 2.5': 1 - self.probabilidades_monte_carlo['over_2_5']
            }
        
        # Caso contrário, calculamos a partir das probabilidades de Poisson
        prob_over_2_5 = 0
        
        for (gols_casa, gols_visitante), prob in self.probabilidades_poisson.items():
            if gols_casa + gols_visitante > 2.5:
                prob_over_2_5 += prob
        
        return {
            'Over 2.5': prob_over_2_5,
            'Under 2.5': 1 - prob_over_2_5
        }
    
    def calcular_probabilidade_btts(self):
        """
        Calcula a probabilidade de ambas as equipes marcarem (BTTS).
        """
        if not self.probabilidades_poisson and not self.probabilidades_monte_carlo:
            self.calcular_probabilidades_poisson()
        
        # Se temos probabilidade de Monte Carlo para BTTS, usamos ela
        if self.probabilidades_monte_carlo and 'btts' in self.probabilidades_monte_carlo:
            return self.probabilidades_monte_carlo['btts']
        
        # Caso contrário, calculamos a partir das probabilidades de Poisson
        prob_btts = 0
        
        for (gols_casa, gols_visitante), prob in self.probabilidades_poisson.items():
            if gols_casa > 0 and gols_visitante > 0:
                prob_btts += prob
        
        return prob_btts
    
    def calcular_valor_esperado(self):
        """
        Calcula o valor esperado (EV) para apostas no resultado.
        
        Retorna:
        dict: Dicionário com o valor esperado para cada resultado
        """
        if not self.odds:
            return {}
        
        probabilidades = self.calcular_probabilidades_resultados()
        
        ev = {}
        if 'casa' in self.odds and self.odds['casa']:
            ev['Casa'] = probabilidades['Casa'] * self.odds['casa'] - 1
        
        if 'empate' in self.odds and self.odds['empate']:
            ev['Empate'] = probabilidades['Empate'] * self.odds['empate'] - 1
        
        if 'visitante' in self.odds and self.odds['visitante']:
            ev['Visitante'] = probabilidades['Visitante'] * self.odds['visitante'] - 1
        
        return ev
    
    def gerar_recomendacoes(self):
        """
        Gera recomendações de apostas com base nas análises.
        
        Retorna:
        list: Lista de recomendações de apostas
        """
        # Se já temos recomendações extraídas do texto científico, retornamos elas
        if self.recomendacoes:
            return self.recomendacoes
        
        recomendacoes = []
        
        # Calcular valor esperado
        ev = self.calcular_valor_esperado()
        melhor_ev = max(ev.items(), key=lambda x: x[1]) if ev else None
        
        # Calcular probabilidades
        prob_over_under = self.calcular_probabilidades_over_under()
        prob_btts = self.calcular_probabilidade_btts()
        
        # Recomendação baseada no melhor valor esperado
        if melhor_ev and melhor_ev[1] > 0:
            recomendacoes.append(f"Aposta com melhor valor esperado: {melhor_ev[0]} (EV: {melhor_ev[1]:.4f})")
        
        # Recomendação baseada em over/under
        if prob_over_under.get('Over 2.5', 0) > 0.6:
            recomendacoes.append(f"Over 2.5 gols (probabilidade: {prob_over_under['Over 2.5']:.2%})")
        elif prob_over_under.get('Under 2.5', 0) > 0.6:
            recomendacoes.append(f"Under 2.5 gols (probabilidade: {prob_over_under['Under 2.5']:.2%})")
        
        # Recomendação baseada em BTTS
        if prob_btts > 0.6:
            recomendacoes.append(f"Ambas equipes marcam - Sim (probabilidade: {prob_btts:.2%})")
        elif (1 - prob_btts) > 0.6:
            recomendacoes.append(f"Ambas equipes marcam - Não (probabilidade: {(1-prob_btts):.2%})")
        
        # Recomendação baseada no placar mais provável
        if self.probabilidades_poisson:
            placar_mais_provavel = list(self.probabilidades_poisson.keys())[0]
            prob_placar = self.probabilidades_poisson[placar_mais_provavel]
            recomendacoes.append(f"Placar exato mais provável: {placar_mais_provavel[0]}x{placar_mais_provavel[1]} (probabilidade: {prob_placar:.2%})")
        
        return recomendacoes
    
    def gerar_analise_completa(self):
        """
        Gera uma análise completa com base nos dados extraídos e cálculos realizados.
        
        Retorna:
        dict: Dicionário com a análise completa
        """
        # Garantir que temos as probabilidades calculadas
        if not self.probabilidades_poisson:
            self.calcular_probabilidades_poisson()
        
        probabilidades_resultados = self.calcular_probabilidades_resultados()
        probabilidades_over_under = self.calcular_probabilidades_over_under()
        probabilidade_btts = self.calcular_probabilidade_btts()
        valor_esperado = self.calcular_valor_esperado()
        recomendacoes = self.gerar_recomendacoes()
        
        # Placares mais prováveis (top 5)
        placares_provaveis = {}
        for i, ((gols_casa, gols_visitante), prob) in enumerate(self.probabilidades_poisson.items()):
            if i < 5:  # Apenas os 5 mais prováveis
                placares_provaveis[f"{gols_casa}x{gols_visitante}"] = prob
            else:
                break
        
        # Montar análise completa
        analise = {
            'times': self.times,
            'odds': self.odds,
            'xg': self.xg,
            'gols_marcados': self.gols_marcados,
            'gols_sofridos': self.gols_sofridos,
            'confrontos_diretos': self.confrontos_diretos,
            'probabilidades_resultados': probabilidades_resultados,
            'probabilidades_over_under': probabilidades_over_under,
            'probabilidade_btts': probabilidade_btts,
            'placares_provaveis': placares_provaveis,
            'valor_esperado': valor_esperado,
            'recomendacoes': recomendacoes
        }
        
        return analise
    
    def gerar_texto_analise_estatistica(self):
        """
        Gera um texto formatado com a análise estatística detalhada.
        
        Retorna:
        str: Texto formatado com a análise estatística
        """
        analise = self.gerar_analise_completa()
        
        texto = f"""
# Análise Estatística: {analise['times']['casa']} x {analise['times']['visitante']}

## Odds e Probabilidades
- Odd {analise['times']['casa']}: {analise['odds'].get('casa', 'N/A')}
- Odd Empate: {analise['odds'].get('empate', 'N/A')}
- Odd {analise['times']['visitante']}: {analise['odds'].get('visitante', 'N/A')}

## Força Ofensiva
- {analise['times']['casa']} (Casa):
  - Gols marcados: {analise['gols_marcados'].get('casa', 'N/A')} por jogo
  - xG médio: {analise['xg'].get('casa', 'N/A')}

- {analise['times']['visitante']} (Fora):
  - Gols marcados: {analise['gols_marcados'].get('visitante', 'N/A')} por jogo
  - xG médio: {analise['xg'].get('visitante', 'N/A')}

## Força Defensiva
- {analise['times']['casa']} (Casa):
  - Gols sofridos: {analise['gols_sofridos'].get('casa', 'N/A')} por jogo

- {analise['times']['visitante']} (Fora):
  - Gols sofridos: {analise['gols_sofridos'].get('visitante', 'N/A')} por jogo

## Confrontos Diretos
- Total de jogos: {analise['confrontos_diretos'].get('total_jogos', 'N/A')}
- Vitórias {analise['times']['casa']}: {analise['confrontos_diretos'].get('vitoria_casa', 'N/A')} ({analise['confrontos_diretos'].get('vitoria_casa_pct', 'N/A')}%)
- Empates: {analise['confrontos_diretos'].get('empates', 'N/A')} ({analise['confrontos_diretos'].get('empate_pct', 'N/A')}%)
- Vitórias {analise['times']['visitante']}: {analise['confrontos_diretos'].get('vitoria_visitante', 'N/A')} ({analise['confrontos_diretos'].get('vitoria_visitante_pct', 'N/A')}%)

## Probabilidades Calculadas
- Vitória {analise['times']['casa']}: {analise['probabilidades_resultados'].get('Casa', 0):.2%}
- Empate: {analise['probabilidades_resultados'].get('Empate', 0):.2%}
- Vitória {analise['times']['visitante']}: {analise['probabilidades_resultados'].get('Visitante', 0):.2%}
- Over 2.5 gols: {analise['probabilidades_over_under'].get('Over 2.5', 0):.2%}
- Under 2.5 gols: {analise['probabilidades_over_under'].get('Under 2.5', 0):.2%}
- Ambas equipes marcam (BTTS): {analise['probabilidade_btts']:.2%}

## Placares Mais Prováveis
"""
        
        # Adicionar placares mais prováveis
        for placar, prob in analise['placares_provaveis'].items():
            texto += f"- {placar}: {prob:.2%}\n"
        
        # Adicionar valor esperado
        texto += "\n## Valor Esperado (EV)\n"
        for resultado, ev in analise['valor_esperado'].items():
            texto += f"- {resultado}: {ev:.4f}\n"
        
        # Adicionar recomendações
        texto += "\n## Recomendações\n"
        for rec in analise['recomendacoes']:
            texto += f"- {rec}\n"
        
        return texto
    
    def gerar_texto_analise_cientifica(self):
        """
        Gera um texto formatado com a análise científica (Poisson, xG, etc.).
        
        Retorna:
        str: Texto formatado com a análise científica
        """
        analise = self.gerar_analise_completa()
        
        texto = f"""🔬 1. Leitura científica com Poisson + xG + odds + valor esperado

📊 Força ofensiva (xG + gols marcados):
{analise['times']['casa']} em casa:
- Gols marcados: {analise['gols_marcados'].get('casa', 'N/A')}
- xG médio: {analise['xg'].get('casa', 'N/A')}

{analise['times']['visitante']} fora:
- Gols marcados: {analise['gols_marcados'].get('visitante', 'N/A')}
- xG médio: {analise['xg'].get('visitante', 'N/A')}

🛡️ Força defensiva (xGA + gols sofridos):
{analise['times']['casa']} em casa:
- Gols sofridos: {analise['gols_sofridos'].get('casa', 'N/A')}

{analise['times']['visitante']} fora:
- Gols sofridos: {analise['gols_sofridos'].get('visitante', 'N/A')}

🔢 Simulação de Poisson (baseada nos xG esperados):
"""
        
        # Adicionar placares mais prováveis
        texto += "Resultado\tProbabilidade\n"
        for placar, prob in analise['placares_provaveis'].items():
            texto += f"{placar}\t{prob:.1%}\n"
        
        # Adicionar placar mais provável
        placar_mais_provavel = list(analise['placares_provaveis'].keys())[0] if analise['placares_provaveis'] else "N/A"
        texto += f"Placar exato mais provável: {placar_mais_provavel}\n\n"
        
        # Adicionar confrontos diretos
        texto += f"""📉 2. Confronto direto
Total de jogos: {analise['confrontos_diretos'].get('total_jogos', 'N/A')}
- Vitórias {analise['times']['casa']}: {analise['confrontos_diretos'].get('vitoria_casa', 'N/A')} ({analise['confrontos_diretos'].get('vitoria_casa_pct', 'N/A')}%)
- Empates: {analise['confrontos_diretos'].get('empates', 'N/A')} ({analise['confrontos_diretos'].get('empate_pct', 'N/A')}%)
- Vitórias {analise['times']['visitante']}: {analise['confrontos_diretos'].get('vitoria_visitante', 'N/A')} ({analise['confrontos_diretos'].get('vitoria_visitante_pct', 'N/A')}%)

"""
        
        # Adicionar simulação Monte Carlo
        texto += f"""🎯 4. Simulação Monte Carlo (10.000 iterações - modelo ajustado Dixon-Coles):
Vitória {analise['times']['casa']}: {analise['probabilidades_resultados'].get('Casa', 0):.1%}
Empate: {analise['probabilidades_resultados'].get('Empate', 0):.1%}
Vitória {analise['times']['visitante']}: {analise['probabilidades_resultados'].get('Visitante', 0):.1%}
Over 2.5 gols: {analise['probabilidades_over_under'].get('Over 2.5', 0):.1%}
BTTS: {analise['probabilidade_btts']:.1%}

"""
        
        # Adicionar mercados com valor esperado
        texto += "💡 5. Mercados com valor esperado (EV):\n"
        for resultado, ev in analise['valor_esperado'].items():
            if ev > 0:
                texto += f"{resultado} → odds justas até {1/analise['probabilidades_resultados'].get(resultado, 0.5):.2f}\n"
        
        # Adicionar recomendações
        texto += "\n✅ 7. Recomendações finais:\n"
        for i, rec in enumerate(analise['recomendacoes']):
            texto += f"🔵 Entrada {i+1}:\n{rec}\n\n"
        
        return texto
    
    def plotar_matriz_poisson(self, max_gols=5):
        """
        Plota uma matriz de calor com as probabilidades de Poisson para diferentes placares.
        
        Parâmetros:
        max_gols (int): Número máximo de gols a considerar
        
        Retorna:
        matplotlib.figure.Figure: Figura com a matriz de calor
        """
        if not self.probabilidades_poisson:
            self.calcular_probabilidades_poisson()
        
        # Criar matriz de probabilidades
        matriz = np.zeros((max_gols + 1, max_gols + 1))
        for i in range(max_gols + 1):
            for j in range(max_gols + 1):
                matriz[i, j] = self.probabilidades_poisson.get((i, j), 0)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, cmap='YlGnBu', fmt='.1%', 
                    xticklabels=range(max_gols + 1), 
                    yticklabels=range(max_gols + 1))
        
        # Configurar rótulos
        ax.set_xlabel(f'Gols {self.times["visitante"]}')
        ax.set_ylabel(f'Gols {self.times["casa"]}')
        ax.set_title('Probabilidades de Placares (Distribuição de Poisson)')
        
        return fig
    
    def plotar_probabilidades_resultados(self):
        """
        Plota um gráfico de barras com as probabilidades de resultados.
        
        Retorna:
        matplotlib.figure.Figure: Figura com o gráfico de barras
        """
        probabilidades = self.calcular_probabilidades_resultados()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Definir cores
        cores = ['#1E88E5', '#FFC107', '#4CAF50']
        
        # Criar gráfico de barras
        resultados = list(probabilidades.keys())
        valores = list(probabilidades.values())
        
        ax.bar(resultados, valores, color=cores)
        
        # Adicionar rótulos
        for i, v in enumerate(valores):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        # Configurar eixos
        ax.set_ylim(0, max(valores) + 0.1)
        ax.set_ylabel('Probabilidade')
        ax.set_title('Probabilidades de Resultados')
        
        return fig
    
    def integrar_com_analisador_apostas(self, analisador_apostas):
        """
        Integra os dados extraídos com o AnalisadorApostas existente.
        
        Parâmetros:
        analisador_apostas: Instância da classe AnalisadorApostas
        
        Retorna:
        bool: True se a integração foi bem-sucedida, False caso contrário
        """
        try:
            # Definir odds alvo se não estiverem definidas
            if analisador_apostas.odds_alvo is None and all(self.odds.values()):
                analisador_apostas.definir_odds_alvo(
                    self.odds['casa'],
                    self.odds['empate'],
                    self.odds['visitante']
                )
            
            # Adicionar análises ao estado do analisador
            if hasattr(analisador_apostas, 'analise_estatistica'):
                analisador_apostas.analise_estatistica = self.gerar_texto_analise_estatistica()
            
            if hasattr(analisador_apostas, 'analise_cientifica'):
                analisador_apostas.analise_cientifica = self.gerar_texto_analise_cientifica()
            
            # Salvar estado
            if hasattr(analisador_apostas, 'salvar_estado'):
                analisador_apostas.salvar_estado()
            
            return True
        except Exception as e:
            print(f"Erro ao integrar com AnalisadorApostas: {str(e)}")
            return False
