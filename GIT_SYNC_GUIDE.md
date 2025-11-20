# Keeping Your Local Folder and GitHub Repo in Sync

## 1. Make Changes Locally
- Edit, add, or delete files in your `candidate-summariser` folder on your laptop.

## 2. Stage Your Changes
Open PowerShell and run:
```powershell
git add .
```
This stages all changes (new, modified, deleted files).

## 3. Commit Your Changes
Run:
```powershell
git commit -m "Describe your changes here"
```
Replace the message in quotes with a short description of what you changed.

## 4. Push Changes to GitHub
Run:
```powershell
git push
```
This uploads your changes to GitHub.

---

## If You Change Files on GitHub (Web)
If you edit files directly on GitHub, run this locally before making new changes:
```powershell
git pull
```
This downloads the latest changes from GitHub to your laptop.

---

## Removing Files/Folders from GitHub (but keeping them locally)
1. Add the file/folder to `.gitignore` (e.g., add `Backup app copies/`)
2. Remove it from GitHub:
   ```powershell
   git rm -r --cached "Backup app copies"
   git commit -m "Remove backup files from repo"
   git push
   ```
This keeps the files on your laptop but removes them from GitHub.

---

## Best Practices
- Always make changes locally, then push to GitHub.
- Avoid editing files directly on GitHub unless necessary.
- Use `git status` to see what’s changed before committing.
- Keep `.gitignore` up to date to avoid uploading unwanted files.

---

## Common Commands Reference
- `git status` — See what’s changed
- `git add .` — Stage all changes
- `git commit -m "message"` — Commit changes
- `git push` — Upload to GitHub
- `git pull` — Download from GitHub
- `git rm -r --cached foldername` — Remove folder from GitHub only

---

If you need more help, ask GitHub Copilot or check this guide!