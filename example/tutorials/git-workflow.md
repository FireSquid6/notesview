# Git Workflow Guide

## Basic Git Commands

### Repository Setup
```bash
# Initialize a new repository
git init

# Clone an existing repository
git clone https://github.com/user/repo.git

# Add remote repository
git remote add origin https://github.com/user/repo.git

# Check remote repositories
git remote -v
```

### Daily Workflow
```bash
# Check status of working directory
git status

# Stage changes
git add .                  # Stage all changes
git add file.txt          # Stage specific file
git add *.js              # Stage all JavaScript files

# Commit changes
git commit -m "Add new feature"
git commit -am "Quick commit of tracked files"

# Push changes
git push origin main
git push -u origin feature-branch  # Set upstream tracking

# Pull changes
git pull origin main
git pull --rebase origin main      # Rebase instead of merge
```

## Branch Management

### Creating and Switching Branches
```bash
# Create new branch
git branch feature-login
git checkout -b feature-login      # Create and switch in one command
git switch -c feature-login        # Modern alternative

# Switch branches
git checkout main
git switch main                    # Modern alternative

# List branches
git branch                         # Local branches
git branch -r                      # Remote branches
git branch -a                      # All branches
```

### Merging and Rebasing
```bash
# Merge feature branch into main
git checkout main
git merge feature-login

# Rebase feature branch onto main
git checkout feature-login
git rebase main

# Interactive rebase (squash commits)
git rebase -i HEAD~3               # Rebase last 3 commits
```

## Advanced Git Operations

### Stashing Changes
```bash
# Stash current changes
git stash
git stash push -m "Work in progress on feature X"

# List stashes
git stash list

# Apply stash
git stash apply                    # Keep stash in list
git stash pop                      # Apply and remove from list
git stash apply stash@{1}          # Apply specific stash

# Drop stash
git stash drop stash@{0}
git stash clear                    # Remove all stashes
```

### Viewing History
```bash
# View commit history
git log
git log --oneline                  # Compact view
git log --graph --oneline          # Visual branch representation
git log --author="John Doe"        # Filter by author
git log --since="2023-01-01"       # Filter by date

# Show changes in commits
git show HEAD                      # Show last commit
git show abc123                    # Show specific commit

# Compare branches/commits
git diff main..feature-branch
git diff HEAD~1 HEAD               # Compare with previous commit
```

### Undoing Changes
```bash
# Undo working directory changes
git checkout -- file.txt          # Restore file from last commit
git restore file.txt               # Modern alternative

# Unstage changes
git reset HEAD file.txt            # Remove from staging area
git restore --staged file.txt      # Modern alternative

# Undo commits
git reset --soft HEAD~1            # Keep changes staged
git reset --mixed HEAD~1           # Keep changes unstaged (default)
git reset --hard HEAD~1            # Discard changes completely

# Revert commits (safe for shared repos)
git revert HEAD                    # Create new commit that undoes last commit
git revert abc123                  # Revert specific commit
```

## Git Flow Workflow

### Feature Development
```bash
# 1. Start new feature from develop
git checkout develop
git pull origin develop
git checkout -b feature/user-authentication

# 2. Work on feature
# ... make changes ...
git add .
git commit -m "Add user registration"
git push -u origin feature/user-authentication

# 3. Create pull request (via GitHub/GitLab UI)
# 4. After review, merge to develop
```

### Release Process
```bash
# 1. Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# 2. Final testing and bug fixes
# ... make fixes ...
git commit -m "Fix critical bug in authentication"

# 3. Merge to main and develop
git checkout main
git merge release/v1.2.0
git tag v1.2.0
git push origin main --tags

git checkout develop
git merge release/v1.2.0
git push origin develop

# 4. Delete release branch
git branch -d release/v1.2.0
git push origin --delete release/v1.2.0
```

### Hotfix Process
```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-security-fix

# 2. Make fix
# ... implement fix ...
git commit -m "Fix security vulnerability in user auth"

# 3. Merge to main and develop
git checkout main
git merge hotfix/critical-security-fix
git tag v1.2.1
git push origin main --tags

git checkout develop
git merge hotfix/critical-security-fix
git push origin develop

# 4. Clean up
git branch -d hotfix/critical-security-fix
git push origin --delete hotfix/critical-security-fix
```

## Collaborative Workflows

### Pull Request Workflow
```bash
# 1. Fork repository (via UI) or create feature branch
git checkout -b feature/improve-performance

# 2. Make changes and commit
git add .
git commit -m "Optimize database queries for better performance"
git push origin feature/improve-performance

# 3. Create pull request (via UI)
# 4. Address review feedback
git add .
git commit -m "Address review comments"
git push origin feature/improve-performance

# 5. After approval, merge and clean up
git checkout main
git pull origin main
git branch -d feature/improve-performance
```

### Resolving Merge Conflicts
```bash
# When merge conflict occurs
git merge feature-branch
# Auto-merging file.txt
# CONFLICT (content): Merge conflict in file.txt
# Automatic merge failed; fix conflicts and then commit the result.

# 1. Edit conflicted files to resolve conflicts
# Look for conflict markers:
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> feature-branch

# 2. Stage resolved files
git add file.txt

# 3. Complete the merge
git commit -m "Resolve merge conflict in file.txt"
```

## Git Configuration

### User Configuration
```bash
# Set user information
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Set default editor
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "vim"          # Vim

# View configuration
git config --list
git config user.name
```

### Aliases
```bash
# Useful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'

# Usage
git st        # Instead of git status
git co main   # Instead of git checkout main
git ci -m "message"  # Instead of git commit -m "message"
```

## Best Practices

### Commit Messages
```
# Good commit message format:
type(scope): brief description

- type: feat, fix, docs, style, refactor, test, chore
- scope: optional component/module affected
- description: imperative mood, present tense

Examples:
feat(auth): add OAuth2 authentication
fix(api): resolve null pointer exception in user service
docs(readme): update installation instructions
refactor(utils): extract validation logic into separate module
```

### Branch Naming
```bash
# Feature branches
feature/user-authentication
feature/payment-integration
feat/shopping-cart

# Bug fix branches
fix/login-error
bugfix/memory-leak
hotfix/security-patch

# Release branches
release/v1.2.0
release/2023-12-15

# Experimental branches
experiment/new-architecture
spike/performance-testing
```

### .gitignore Examples
```gitignore
# Dependencies
node_modules/
*.log

# Build outputs
dist/
build/
*.min.js
*.min.css

# Environment files
.env
.env.local
.env.production

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Language specific
__pycache__/
*.pyc
*.class
target/
.gradle/
```

## Git Hooks

### Pre-commit Hook Example
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run linting
npm run lint
if [ $? -ne 0 ]; then
  echo "Linting failed. Please fix errors before committing."
  exit 1
fi

# Run tests
npm test
if [ $? -ne 0 ]; then
  echo "Tests failed. Please fix tests before committing."
  exit 1
fi

echo "Pre-commit checks passed!"
```

### Commit-msg Hook Example
```bash
#!/bin/sh
# .git/hooks/commit-msg

commit_regex='^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Format: type(scope): description"
    echo "Example: feat(auth): add login functionality"
    exit 1
fi
```

## Troubleshooting

### Common Issues
```bash
# Accidentally committed to wrong branch
git reset --soft HEAD~1       # Undo commit, keep changes
git stash                      # Stash changes
git checkout correct-branch    # Switch to correct branch
git stash pop                  # Apply changes
git commit -m "Correct commit message"

# Remove file from Git but keep locally
git rm --cached file.txt
echo "file.txt" >> .gitignore

# Sync fork with upstream
git remote add upstream https://github.com/original/repo.git
git fetch upstream
git checkout main
git rebase upstream/main
git push origin main --force-with-lease

# Clean up local branches
git branch --merged main | grep -v "main" | xargs -n 1 git branch -d
git remote prune origin        # Remove tracking branches for deleted remotes
```