use crate::types::Metadata;

/// Metadata filter expressions (mirrors Python SDK semantics).
#[derive(Clone, Debug)]
pub enum Filter {
    Eq {
        field: String,
        value: Metadata,
    },
    Lt {
        field: String,
        value: Metadata,
    },
    Gt {
        field: String,
        value: Metadata,
    },
    Lte {
        field: String,
        value: Metadata,
    },
    Gte {
        field: String,
        value: Metadata,
    },
    Ne {
        field: String,
        value: Metadata,
    },
    In {
        field: String,
        values: Vec<Metadata>,
    },
    Nin {
        field: String,
        values: Vec<Metadata>,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Document filter expressions.
#[derive(Clone, Debug)]
pub enum DocFilter {
    Contains(String),
    Regex(String),
    And(Vec<DocFilter>),
    Or(Vec<DocFilter>),
}

/// Helper for SQL WHERE clause + bound parameters.
#[derive(Clone, Debug)]
pub struct SqlWhere {
    pub clause: String,
    pub params: Vec<Metadata>,
}

/// Build SQL WHERE clause from metadata/doc filters and optional ids.
/// Mirrors the Python client's `_build_where_clause` and `FilterBuilder`.
pub fn build_where_clause(
    filter: Option<&Filter>,
    doc_filter: Option<&DocFilter>,
    ids: Option<&[String]>,
) -> SqlWhere {
    let mut clauses: Vec<String> = Vec::new();
    let mut params: Vec<Metadata> = Vec::new();

    // IDs filter: generate `_id IN (?, ?, ...)`
    if let Some(ids) = ids {
        if !ids.is_empty() {
            let placeholders = std::iter::repeat("?")
                .take(ids.len())
                .collect::<Vec<_>>()
                .join(", ");
            clauses.push(format!("_id IN ({placeholders})"));
            for id in ids {
                params.push(Metadata::String(id.clone()));
            }
        }
    }

    // Metadata filter
    if let Some(filter) = filter {
        let (clause, mut p) = build_meta_clause(filter);
        if !clause.is_empty() {
            clauses.push(clause);
            params.append(&mut p);
        }
    }

    // Document filter
    if let Some(doc_filter) = doc_filter {
        let (clause, mut p) = build_doc_clause(doc_filter);
        if !clause.is_empty() {
            clauses.push(clause);
            params.append(&mut p);
        }
    }

    let clause = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };

    SqlWhere { clause, params }
}

fn build_meta_clause(filter: &Filter) -> (String, Vec<Metadata>) {
    let mut params = Vec::new();
    let clause = match filter {
        Filter::Eq { field, value } => {
            params.push(value.clone());
            format!("JSON_EXTRACT(metadata, '$.{field}') = ?")
        }
        Filter::Lt { field, value } => {
            params.push(value.clone());
            format!("JSON_EXTRACT(metadata, '$.{field}') < ?")
        }
        Filter::Gt { field, value } => {
            params.push(value.clone());
            format!("JSON_EXTRACT(metadata, '$.{field}') > ?")
        }
        Filter::Lte { field, value } => {
            params.push(value.clone());
            format!("JSON_EXTRACT(metadata, '$.{field}') <= ?")
        }
        Filter::Gte { field, value } => {
            params.push(value.clone());
            format!("JSON_EXTRACT(metadata, '$.{field}') >= ?")
        }
        Filter::Ne { field, value } => {
            params.push(value.clone());
            format!("JSON_EXTRACT(metadata, '$.{field}') != ?")
        }
        Filter::In { field, values } => {
            let placeholders = std::iter::repeat("?")
                .take(values.len())
                .collect::<Vec<_>>()
                .join(", ");
            params.extend(values.iter().cloned());
            format!("JSON_EXTRACT(metadata, '$.{field}') IN ({placeholders})")
        }
        Filter::Nin { field, values } => {
            let placeholders = std::iter::repeat("?")
                .take(values.len())
                .collect::<Vec<_>>()
                .join(", ");
            params.extend(values.iter().cloned());
            format!("JSON_EXTRACT(metadata, '$.{field}') NOT IN ({placeholders})")
        }
        Filter::And(filters) => {
            let mut clauses = Vec::new();
            for f in filters {
                let (c, mut p) = build_meta_clause(f);
                if !c.is_empty() {
                    clauses.push(c);
                    params.append(&mut p);
                }
            }
            format!("({})", clauses.join(" AND "))
        }
        Filter::Or(filters) => {
            let mut clauses = Vec::new();
            for f in filters {
                let (c, mut p) = build_meta_clause(f);
                if !c.is_empty() {
                    clauses.push(c);
                    params.append(&mut p);
                }
            }
            format!("({})", clauses.join(" OR "))
        }
        Filter::Not(f) => {
            let (c, mut p) = build_meta_clause(f);
            if !c.is_empty() {
                params.append(&mut p);
                format!("NOT ({c})")
            } else {
                String::new()
            }
        }
    };

    (clause, params)
}

fn build_doc_clause(filter: &DocFilter) -> (String, Vec<Metadata>) {
    let mut params = Vec::new();
    let clause = match filter {
        DocFilter::Contains(text) => {
            params.push(Metadata::String(text.clone()));
            "MATCH(document) AGAINST (? IN NATURAL LANGUAGE MODE)".to_string()
        }
        DocFilter::Regex(pattern) => {
            params.push(Metadata::String(pattern.clone()));
            "document REGEXP ?".to_string()
        }
        DocFilter::And(filters) => {
            let mut clauses = Vec::new();
            for f in filters {
                let (c, mut p) = build_doc_clause(f);
                if !c.is_empty() {
                    clauses.push(c);
                    params.append(&mut p);
                }
            }
            format!("({})", clauses.join(" AND "))
        }
        DocFilter::Or(filters) => {
            let mut clauses = Vec::new();
            for f in filters {
                let (c, mut p) = build_doc_clause(f);
                if !c.is_empty() {
                    clauses.push(c);
                    params.append(&mut p);
                }
            }
            format!("({})", clauses.join(" OR "))
        }
    };

    (clause, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_where_clause_with_ids_meta_doc() {
        let filter = Filter::Gte {
            field: "age".into(),
            value: json!(18),
        };
        let doc = DocFilter::Contains("hello".into());
        let ids = vec!["1".into(), "2".into(), "3".into()];

        let sql = build_where_clause(Some(&filter), Some(&doc), Some(&ids));
        assert_eq!(
            sql.clause,
            "WHERE _id IN (?, ?, ?) AND JSON_EXTRACT(metadata, '$.age') >= ? AND MATCH(document) AGAINST (? IN NATURAL LANGUAGE MODE)"
        );
        assert_eq!(
            sql.params,
            vec![
                json!("1"),
                json!("2"),
                json!("3"),
                json!(18),
                json!("hello")
            ]
        );
    }

    #[test]
    fn test_doc_regex_or() {
        let doc = DocFilter::Or(vec![
            DocFilter::Regex("^a.*".into()),
            DocFilter::Regex("b$".into()),
        ]);
        let sql = build_where_clause(None, Some(&doc), None);
        assert_eq!(sql.clause, "WHERE (document REGEXP ? OR document REGEXP ?)");
        assert_eq!(sql.params, vec![json!("^a.*"), json!("b$")]);
    }
}
