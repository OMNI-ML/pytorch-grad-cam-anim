# ref -> https://stackoverflow.com/questions/46704572/git-error-encountered-7-files-that-should-have-been-pointers-but-werent

git rm --cached -r .
git reset --hard
git rm .gitattributes
git reset .
git checkout .