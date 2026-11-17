# AP7 — Classificação de mensagens da UniAssist

**Disciplina:** Inteligência Artificial · **Aula:** Processamento de Linguagens Naturais  
**Modalidade:** individual ou dupla · **Duração:** aproximadamente 90–110 minutos.

## Contexto
A central de atendimento da UniAssist recebe pedidos destinados a **Financeiro, Matrícula, Estágio e Suporte**. Há 80 mensagens acadêmicas fictícias, previamente classificadas por uma equipe, em `mensagens.csv`. Sua tarefa é comparar duas formas de encaminhamento: **regras explícitas** e **aprendizado supervisionado com TF-IDF + SVM linear**.

O objetivo é construir um protótipo reproduzível e identificar limites, principalmente diante de mensagens ambíguas.

## Preparação do ambiente (antes da aula, se estiver offline)
- Python 3 e VS Code com extensão Python instalados.
- Pacote `scikit-learn` instalado previamente com `python -m pip install -r requirements.txt` enquanto houver conexão. Não será necessário baixar modelos ou dados durante a aula.
- Abrir esta pasta no VS Code. Os CSVs são locais e utilizam UTF-8.

## Arquivos fornecidos
- `mensagens.csv`: **80 exemplos sintéticos**, balanceados em quatro categorias (20 cada), com colunas `texto,categoria`.
- `desafios.csv`: **8 mensagens novas** com ambiguidade e vocabulário diferente; sem rótulo de resposta.
- `uniaassist_pln.py`: código inicial com seis TODOs.

## Missão
1. **Leia os dados.** Verifique quantas categorias existem e quantos exemplos há em cada uma.
2. **Construa um baseline por regras** conforme `TODO 1`, contabilizando correspondências; em empate ou ausência de termos, devolva `Revisão humana`.
3. **Divida treino e teste** conforme `TODO 2`. Não ajuste TF-IDF usando todo o dataset, para não vazar informações do teste.
4. **Treine o pipeline TF-IDF + LinearSVC** e apresente a acurácia, classificação por classe e matriz de confusão no teste.
5. **Compare com regras no mesmo conjunto de teste.** Calcule a proporção de mensagens classificadas corretamente; `Revisão humana` conta como não acerto automatizado.
6. **Aplique os dois métodos às oito mensagens de `desafios.csv`.** Registre diferenças, casos ambíguos e quando o atendimento humano seria mais apropriado.
7. **Experimento de melhoria:** altere `ngram_range=(1, 2)` para `(1, 1)` e compare métricas. Não escolha um vencedor apenas com base no teste; explique por que esse uso repetido do teste é metodologicamente limitado.

## Entrega obrigatória
Entregue somente:
- `uniaassist_pln.py` — completo e executável, sem `NotImplementedError`;
- `parecer.md` — tabela com acurácia do classificador e do baseline, matriz de confusão, observações de pelo menos 3 mensagens de desafio e um parecer técnico de até 15 linhas sobre limitações.

## Restrições
Sem internet, APIs, IA generativa ou modelos baixados durante a aula. Não altere os rótulos do dataset para melhorar métricas. Respeite a divisão treino/teste definida. Evite apresentar a saída do classificador como uma decisão infalível.

## Critérios de correção
Funcionamento e reprodutibilidade (30%), preparação correta do treino/teste e pipeline (25%), interpretação de métricas (20%), comparação com regras e análise das ambiguidades (25%).
