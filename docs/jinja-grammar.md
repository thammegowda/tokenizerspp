# Jinja2 Grammar for Chat Templates (tokenizerpp)

## Notation
- `|` = alternatives
- `[...]` = optional
- `{...}` = zero or more repetitions  
- `(...)` = grouping
- `'...'` = literal terminal
- `UPPER` = token class

## 1. Template Structure

```
template       ::= { text | comment | variable_tag | block_tag }

text           ::= (any chars not starting a tag)

comment        ::= '{#' ['-'] ... ['-'] '#}'

variable_tag   ::= '{{' ['-'] expression ['-'] '}}'

block_tag      ::= '{%' ['-'] statement ['-'] '%}'
```

Whitespace control: a `-` immediately after `{%`/`{{` or before `%}`/`}}` strips adjacent whitespace.

## 2. Statements

```
statement      ::= for_stmt
                 | endfor_stmt
                 | if_stmt
                 | elif_stmt
                 | else_stmt
                 | endif_stmt
                 | set_stmt
                 | macro_stmt
                 | endmacro_stmt
                 | raw_stmt
                 | endraw_stmt

for_stmt       ::= 'for' target_list 'in' expression [ 'if' expression ] [ 'recursive' ]
endfor_stmt    ::= 'endfor'

if_stmt        ::= 'if' expression
elif_stmt      ::= 'elif' expression
else_stmt      ::= 'else'
endif_stmt     ::= 'endif'

set_stmt       ::= 'set' target '=' expression      -- inline assignment
                 | 'set' target                     -- block: {% set x %}...{% endset %}
endset_stmt    ::= 'endset'

macro_stmt     ::= 'macro' IDENT '(' [ param_list ] ')'
endmacro_stmt  ::= 'endmacro'

target_list    ::= target { ',' target }
target         ::= IDENT | IDENT '.' IDENT    -- namespace attribute assignment
```

## 3. Expressions (precedence low→high)

```
expression     ::= conditional_expr

conditional_expr ::= or_expr [ 'if' or_expr [ 'else' expression ] ]

or_expr        ::= and_expr { 'or' and_expr }

and_expr       ::= not_expr { 'and' not_expr }

not_expr       ::= 'not' not_expr
                 | compare_expr

compare_expr   ::= concat_expr { comp_op concat_expr }
comp_op        ::= '==' | '!=' | '<' | '>' | '<=' | '>='
                 | 'in' | 'not' 'in'
                 | 'is' test_name
                 | 'is' 'not' test_name

concat_expr    ::= add_expr { '~' add_expr }

add_expr       ::= mul_expr { ('+' | '-') mul_expr }

mul_expr       ::= pow_expr { ('*' | '/' | '//' | '%') pow_expr }

pow_expr       ::= unary_expr [ '**' unary_expr ]

unary_expr     ::= ('-' | '+') unary_expr
                 | postfix_expr

postfix_expr   ::= primary { postfix_op }
postfix_op     ::= '.' IDENT                     -- attribute access
                 | '[' expression ']'             -- subscript
                 | '[' [expression] ':' [expression] [ ':' [expression] ] ']'  -- slice
                 | '(' [arg_list] ')'             -- function call
                 | '|' IDENT [ '(' [arg_list] ')' ]       -- filter

primary        ::= IDENT
                 | STRING_LITERAL
                 | INTEGER_LITERAL
                 | FLOAT_LITERAL
                 | 'true' | 'True'
                 | 'false' | 'False'
                 | 'none' | 'None'
                 | '[' [expression_list] ']'      -- list literal
                 | '(' [expression_list] ')'      -- tuple / grouping
                 | '{' [dict_items] '}'           -- dict literal

expression_list ::= expression { ',' expression } [',']
dict_items     ::= dict_item { ',' dict_item } [',']
dict_item      ::= expression ':' expression

arg_list       ::= arg { ',' arg }
arg            ::= expression | IDENT '=' expression    -- positional or keyword

param_list     ::= param { ',' param }
param          ::= IDENT [ '=' expression ]             -- with optional default
```

## 4. Tests

```
test_name      ::= 'defined' | 'undefined'
                 | 'none'
                 | 'true' | 'false'
                 | 'string' | 'number' | 'integer' | 'float'
                 | 'mapping' | 'iterable' | 'sequence'
                 | 'callable'
                 | 'even' | 'odd'
                 | 'divisibleby' [ '(' expression ')' ]
                 | 'eq' | 'equalto' | 'ne'
                 | 'gt' | 'ge' | 'lt' | 'le'
                 | 'sameas'
                 | 'in'
                 | 'lower' | 'upper'
```

Usage: `expression 'is' ['not'] test_name [ '(' arg_list ')' ]`

## 5. Builtin Filters

### Critical (used in real chat templates)
```
| length                              -- len(value)
| trim                                -- strip whitespace
| default(value, boolean=false)       -- fallback (alias: d)
| join(separator='')                  -- join list
| first                               -- first element
| last                                -- last element
| replace(old, new[, count])          -- string replace
| upper                               -- uppercase
| lower                               -- lowercase
| title                               -- title case
| tojson                              -- JSON serialize
| int(default=0)                      -- to integer
| float(default=0.0)                  -- to float
| string                              -- to string
| list                                -- to list
| reverse                             -- reverse sequence
| selectattr(attr[, test[, value]])   -- filter by attribute
| map(attribute=name)                 -- extract attribute
| sort(reverse=false)                 -- sort list
| batch(count, fill_with=none)        -- batch into groups
| reject(test)                        -- reject matching
| rejectattr(attr[, test[, value]])   -- reject by attribute
| select(test)                        -- select matching
```

### Lower priority
```
| abs | attr | capitalize | center | dictsort | escape (e)
| filesizeformat | format | groupby | indent | max | min
| pprint | random | round | safe | slice | striptags
| sum | truncate | unique | urlencode | urlize
| wordcount | wordwrap | xmlattr
```

## 6. Global Functions

```
range([start,] stop[, step])
namespace(**kwargs)                   -- mutable state object
raise_exception(message)              -- custom: raise error
strftime_now(format)                  -- custom: current datetime
```

## 7. Lexical Elements

```
IDENT          ::= [a-zA-Z_][a-zA-Z0-9_]*
STRING_LITERAL ::= '"' {char | escape} '"' | "'" {char | escape} "'"
escape         ::= '\\' ('\\' | '"' | "'" | 'n' | 't' | 'r' | 'x' HEX HEX | 'u' HEX{4})
INTEGER_LITERAL ::= DIGIT { DIGIT | '_' }
FLOAT_LITERAL  ::= DIGIT { DIGIT } '.' { DIGIT } [ ('e'|'E') ['+' | '-'] DIGIT { DIGIT } ]
```

### Reserved keywords
```
and  as  block  else  elif  endblock  endfor  endif  endmacro
endraw  endset  extends  false  filter  for  from  if  ignore
import  in  include  is  macro  missing  none  not  or  raw
recursive  scoped  set  true  with  without
```

## 8. Scoping Rules

1. **Top level**: variables visible everywhere in template
2. **For loops**: create new scope — variables set inside are NOT visible outside
3. **If blocks**: do NOT create a new scope — variables set inside ARE visible
4. **Namespace objects**: allow mutation across scopes:
   ```jinja
   {% set ns = namespace(found=false) %}
   {% for item in items %}
       {% if item.match %}{% set ns.found = true %}{% endif %}
   {% endfor %}
   {{ ns.found }}  {# visible here! #}
   ```
5. **`set` with `IDENT.attr`** syntax only works on namespace objects

## 9. Loop Variables

Inside `{% for %}` blocks:

| Variable | Type | Description |
|----------|------|-------------|
| `loop.index` | int | Current iteration (1-indexed) |
| `loop.index0` | int | Current iteration (0-indexed) |
| `loop.revindex` | int | Iterations from end (1-indexed) |
| `loop.revindex0` | int | Iterations from end (0-indexed) |
| `loop.first` | bool | True if first iteration |
| `loop.last` | bool | True if last iteration |
| `loop.length` | int | Total items in sequence |
| `loop.previtem` | any | Previous iteration item |
| `loop.nextitem` | any | Next iteration item |
| `loop.depth` | int | Nesting depth (starts at 1) |
| `loop.depth0` | int | Nesting depth (starts at 0) |
| `loop.cycle(...)` | func | Cycle through values |
| `loop.changed(val)` | bool | Value changed since last call |

## 10. Operator Precedence (low → high)

| Prec | Operators | Associativity |
|------|-----------|---------------|
| 1 | `if ... else` (ternary) | right |
| 2 | `or` | left |
| 3 | `and` | left |
| 4 | `not` | prefix |
| 5 | `==  !=  <  >  <=  >=  in  not in  is  is not` | left |
| 6 | `~` (concat) | left |
| 7 | `+  -` | left |
| 8 | `*  /  //  %` | left |
| 9 | `**` | left (NOTE: Jinja is left-to-right, unlike Python) |
| 10 | `- +` (unary) | prefix |
| 11 | `.  []  ()  \|` (postfix) | left |

## 11. Implementation Status (tokenizerpp)

### ✅ Implemented
- All expression types, literals, operators
- For/if/elif/else/set/endfor/endif/endset
- Block set: `{% set x %}...{% endset %}` (captures rendered body)
- Macros: `{% macro name(p, k=default) %}...{% endmacro %}` with positional
  args, keyword args, default values, and recursion (capped at 64 nested calls,
  beyond which rendering fails with an error). Macros push a scope onto
  the current stack, so they can read globals and mutate enclosing `namespace`
  objects. Invoked via `{{ name(args) }}` (result is the rendered string).
- Filters: length, trim, default, first, last, upper, lower, title, join,
  replace, tojson, int, float, string, list, map, map('filter'), selectattr,
  batch, reverse, dictsort (case-insensitive unless `dictsort(true)`), items
- Tests: defined, undefined, none, string, number, integer, float, iterable,
  sequence, mapping, callable, boolean, true, false, even, odd, eq/ne
- Functions: raise_exception, namespace, range, dict
- String/object methods (.upper, .lower, .strip, .lstrip, .rstrip, .split,
  .replace, .startswith, .endswith, .items, .keys, .values, .get)
- Whitespace control, comments
- List slicing with step (e.g. `xs[::-1]`, `xs[1:4]`, `xs[::2]`), dict
  literals, negative indexing. Out-of-range bounds and steps are clamped to the
  sequence length, as in Python.
- Loop variables incl. `loop.previtem` / `loop.nextitem` (null at boundaries)
- Undefined variables are falsy (`{% if missing %}` → false)

### ⬜ Not needed for chat templates
- Template inheritance, import, include
- Recursive loops (`{% for ... recursive %}`), filter blocks, `{% call %}`
- HTML escaping
