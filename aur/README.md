# AUR Package for mdserve

This directory contains the files needed to publish `mdserve` to the Arch User Repository (AUR).

## Package Variants

- **PKGBUILD** - Binary package (`mdserve-bin`): Downloads pre-built binary from GitHub releases
- **PKGBUILD-git** - Source package (`mdserve`): Builds from source using Bun

## Prerequisites

1. An Arch Linux system (or use a VM/container)
2. An AUR account: https://aur.archlinux.org/register
3. SSH key added to your AUR account

## Publishing to AUR

You can publish either or both packages to AUR. The binary package is easier for users (no build dependencies), while the source package gives users more control and transparency.

### Publishing mdserve-bin (Binary Package)

#### First-time setup

1. **Update the PKGBUILD maintainer info:**
   Edit `PKGBUILD` and replace `Your Name <your.email@example.com>` with your actual info.

2. **Create a release on GitHub:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
   This will trigger the GitHub Action to build binaries and create a release.

3. **Update the checksum:**
   After the release is created, download the `mdserve-linux-x64` binary and calculate its SHA256:
   ```bash
   sha256sum mdserve-linux-x64
   ```
   Update the `sha256sums_x86_64` in PKGBUILD with the actual checksum (replace 'SKIP').

4. **Test the package locally:**
   ```bash
   cd aur
   makepkg -si
   ```
   This will build and install the package locally to verify it works.

5. **Generate .SRCINFO:**
   ```bash
   makepkg --printsrcinfo > .SRCINFO
   ```

6. **Clone the AUR repository:**
   ```bash
   git clone ssh://aur@aur.archlinux.org/mdserve-bin.git aur-repo
   cd aur-repo
   ```

7. **Copy files and push:**
   ```bash
   cp ../PKGBUILD ../.SRCINFO .
   git add PKGBUILD .SRCINFO
   git commit -m "Initial commit: mdserve-bin 1.0.0"
   git push
   ```

#### Updating the package

When you release a new version:

1. **Create a new GitHub release** (tag it with the version, e.g., `v1.1.0`)

2. **Update PKGBUILD:**
   - Update `pkgver` to the new version
   - Reset `pkgrel` to 1
   - Update `sha256sums_x86_64` with the new checksum

3. **Test locally:**
   ```bash
   makepkg -si
   ```

4. **Update .SRCINFO:**
   ```bash
   makepkg --printsrcinfo > .SRCINFO
   ```

5. **Push to AUR:**
   ```bash
   cd aur-repo
   cp ../PKGBUILD ../.SRCINFO .
   git add PKGBUILD .SRCINFO
   git commit -m "Update to version X.Y.Z"
   git push
   ```

### Publishing mdserve (Source Package)

#### First-time setup

1. **Create a release on GitHub:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Update the checksum:**
   Download the source tarball and calculate its SHA256:
   ```bash
   curl -L -o mdserve-1.0.0.tar.gz https://github.com/FireSquid6/markdown-preview-server/archive/v1.0.0.tar.gz
   sha256sum mdserve-1.0.0.tar.gz
   ```
   Update the `sha256sums` in PKGBUILD-git with the actual checksum.

3. **Test the package locally:**
   ```bash
   cd aur
   makepkg -p PKGBUILD-git -si
   ```

4. **Generate .SRCINFO:**
   ```bash
   makepkg -p PKGBUILD-git --printsrcinfo > .SRCINFO
   ```

5. **Clone the AUR repository:**
   ```bash
   git clone ssh://aur@aur.archlinux.org/mdserve.git mdserve-repo
   cd mdserve-repo
   ```

6. **Copy files and push:**
   ```bash
   cp ../PKGBUILD-git PKGBUILD
   cp ../.SRCINFO .
   git add PKGBUILD .SRCINFO
   git commit -m "Initial commit: mdserve 1.0.0"
   git push
   ```

#### Updating the package

When you release a new version:

1. **Create a new GitHub release** (tag it with the version, e.g., `v1.1.0`)

2. **Update PKGBUILD-git:**
   - Update `pkgver` to the new version
   - Reset `pkgrel` to 1
   - Update `sha256sums` with the new source tarball checksum

3. **Test locally:**
   ```bash
   makepkg -p PKGBUILD-git -si
   ```

4. **Update .SRCINFO:**
   ```bash
   makepkg -p PKGBUILD-git --printsrcinfo > .SRCINFO
   ```

5. **Push to AUR:**
   ```bash
   cd mdserve-repo
   cp ../PKGBUILD-git PKGBUILD
   cp ../.SRCINFO .
   git add PKGBUILD .SRCINFO
   git commit -m "Update to version X.Y.Z"
   git push
   ```

## Installation (for users)

Once published to AUR, users can install with:

**Binary package (recommended for most users):**
```bash
# Using yay
yay -S mdserve-bin

# Using paru
paru -S mdserve-bin

# Manual installation
git clone https://aur.archlinux.org/mdserve-bin.git
cd mdserve-bin
makepkg -si
```

**Source package (requires Bun):**
```bash
# Using yay
yay -S mdserve

# Using paru
paru -S mdserve

# Manual installation
git clone https://aur.archlinux.org/mdserve.git
cd mdserve
makepkg -si
```

## Resources

- [AUR Submission Guidelines](https://wiki.archlinux.org/title/AUR_submission_guidelines)
- [PKGBUILD Reference](https://wiki.archlinux.org/title/PKGBUILD)
- [Creating Packages](https://wiki.archlinux.org/title/Creating_packages)
