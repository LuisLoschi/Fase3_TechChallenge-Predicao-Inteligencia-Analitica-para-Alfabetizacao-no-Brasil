# Contrato temporal e de uso

**CRISP-DM: Business Understanding / Data Preparation.** A decisão atual é apoiar
análise retrospectiva de territórios. O usuário é a equipe de planejamento educacional.
O escore não diagnostica crianças e não autoriza intervenção individual automatizada.

Classe positiva: não alfabetizado em 2024, entre alunos presentes com medida válida.
Falso positivo pode desviar capacidade de análise; falso negativo deixa de priorizar um contexto de risco.
Não foram atribuídos custos monetários. A capacidade de 20% é uma hipótese operacional,
não um orçamento aprovado por secretaria. O critério técnico é comparar com baselines
nos mesmos grupos; adoção prospectiva exige evidência adicional e custos acordados.

Momento hipotético de previsão: após divulgação dos dados de 2023 e antes da avaliação de 2024.
Esse cenário **não foi comprovado**: o snapshot foi extraído em 2026 e não contém
um calendário auditado de disponibilidade. Ano de referência não equivale a data de publicação.
Território e rede podem ser conhecidos por matrícula, mas essa disponibilidade é uma premissa.
O caderno é da prova de 2024 e só cabe no artefato retrospectivo até comprovação contrária.

| Atributo | Referência/origem efetiva | Disponibilidade antes da prova de 2024 |
|---|---|---|
| `mun_taxa_alfab_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `mun_media_portugues_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `mun_nivel_alfabetizacao_lag1` | linha 2023 da tabela de metas | publicação e revisões não verificadas |
| `mun_meta_2024_lag1` | linha 2023 da tabela de metas | publicação e revisões não verificadas |
| `mun_taxa_presenca_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `mun_n_alunos_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `mun_n_escolas_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `mun_share_rede_estadual_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `mun_desvio_vs_uf` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `uf_taxa_alfab_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `uf_media_portugues_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `uf_taxa_presenca_lag1` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `tem_historico_municipio` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `tem_historico_escola` | interseção de IDs 2023/2024 reciclados | sem significado longitudinal; artefato retrospectivo |
| `fonte_lag_municipal` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `fonte_lag_uf` | apuração de 2023 ou proveniência do join | publicação não verificada |
| `rede_grupo` | cadastro da coorte de 2024 | premissa de matrícula; não comprovada |
| `nome_regiao` | cadastro da coorte de 2024 | premissa de matrícula; não comprovada |
| `sigla_uf` | cadastro da coorte de 2024 | premissa de matrícula; não comprovada |
| `caderno` | caderno de prova de 2024 | não comprovada; controle retrospectivo |


Não foi identificado atributo socioeconômico. O enunciado menciona essa dimensão;
a lacuna permanece até integração de fonte adequada ou alinhamento acadêmico do recorte.
Não se supõe aceite do professor. Sem nova amostra intocada, não há gate de validação
confirmatória. As análises deste trabalho são exploratórias.
