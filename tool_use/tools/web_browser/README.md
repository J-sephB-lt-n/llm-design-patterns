# Web Browser Tools

## The current plan

```python
>>> go_to_url("https://www.google.com")
found 12 <h1> tags
found 99 buttons
found 27 select inputs
found 4 text inputs
found 434 hyperlinks
'<shows page 1/400 of markdownified page content here>'
>>> scroll_to_page(page_num=2)
'<shows text of page 2/400>'
>>> show_available_hyperlinks(regex_pattern='...')
'<list of hyperlinks matching given pattern>'
>>> full_text_search(regex_pattern='...')
Found match in 4 locations.
Here is match location 1/4:
'<shows text containing first match>'
>>> full_text_search(regex_pattern='...', match_num=2)
Found match in 4 locations.
Here is match location 2/4:
'<shows text containing second match>'
>>> go_to_url(...)
```
