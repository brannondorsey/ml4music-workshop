<input.txt> was created by downloading and combining lyrics from 4,801 hip-hop songs from the [Original Hip-Hop Lyrics Archive](http://ohhla.com/).
Lyrics were scrapped with the following commands:

```bash
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent -N http://ohhla.com/all.html
```
This command can take hours to run. Consistently check to see how many lyrics files have been downloaded with:

```bash
finx ohhla.com -name *.txt | wc -l
```

If that number stops going up for a long period of time (overnight) you can likely stop the download. Combine all lyrics into one file with:

```bash
find ohhla.com -name *.txt | xargs cat > lyrics_tmp.txt
```

```bash
# strip non utf-8 chars
iconv -f utf-8 -t utf-8 -c lyrics_tmp.txt -o lyrics.txt
```