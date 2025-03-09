```markdown
# Security Code Review Standardization Task

## Your Role
You are a code reviewer. Your task is to optimize code review comments into a standardized security review format, highlighting potential security issues.

## Output Format
The output should be in JSON format with specific fields to ensure consistency and thoroughness. The JSON object must include:

- **Description**: Clearly explain the security issue found in the provided code change.
- **Impact**: Highlight the potential security consequences if the issue is left unresolved.
- **Advice**: Offer recommendations for resolving the issue.

## Example

### Code Change:
```diff
func Batch(chunkSize uint, cb func(offset, limit uint) (uint, error)) error {
    offset += limit
}

+func (orm *ORM) rowExists(query string, args ...interface{}) (bool, error) {
+   var exists bool
+   query = fmt.Sprintf("SELECT exists (%s)", query)
+   err := orm.db.DB().QueryRow(query, args...).Scan(&exists)
+   if err != nil && err != sql.ErrNoRows {
+       return false, err
+   }
+   return exists, nil
+}
```

### Original Comment:
"SQL injection attack vector."

### Security Type:
Input Validation

### Standard Review Comments:
```json
{
  "description": "The new `rowExists` function in the ORM layer constructs a SQL query by embedding an unvalidated `query` string directly into a SQL command. This approach can lead to SQL injection vulnerabilities if the `query` string contains untrusted input.",
  "impact": "SQL injection vulnerabilities can allow attackers to manipulate queries to access, modify, or delete data arbitrarily, bypass application logic, escalate privileges, or execute administrative operations on the database. This can lead to data breaches, loss of data integrity, and unauthorized access to sensitive information within the database.",
  "advice": "Use parameterized queries or prepared statements to construct SQL queries safely. For the `rowExists` function, refactor the query construction to avoid directly embedding the `query` string into the SQL command. For example, modify the function to accept only parameters for conditions rather than entire query fragments, or ensure that any input used to construct queries is strictly validated and sanitized before use."
}
```

## Your Task
Process the following data according to the instructions and example:

### Code Change:
{}

### Original Comment:
{}

### Security Type:
{}

### Standard Review Comments:
```