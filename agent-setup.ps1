# Recreate the local Claude skill view after clone; canonical content stays in .agents.
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$canonical = Join-Path $repo '.agents/skills'
$view = Join-Path $repo '.claude/skills'
if (-not (Test-Path -LiteralPath $canonical)) { Write-Output 'No project skills to link.'; exit 0 }
if (Test-Path -LiteralPath $view) {
    $item = Get-Item -LiteralPath $view -Force
    if ($item.LinkType -ne 'Junction' -and $item.LinkType -ne 'SymbolicLink') {
        $children = @(Get-ChildItem -LiteralPath $view -Force)
        $sources = @(Get-ChildItem -LiteralPath $canonical -Force)
        if ($children.Count -eq 0 -or $children.Count -ne $sources.Count) { throw 'Existing real .claude/skills: reconcile it before setup.' }
        foreach ($child in $children) {
            if ($child.LinkType -ne 'Junction' -and $child.LinkType -ne 'SymbolicLink') { throw 'Unlinked skill in Claude view.' }
            $expected = Join-Path $canonical $child.Name
            if ((Resolve-Path -LiteralPath $child.Target).Path -ne (Resolve-Path -LiteralPath $expected).Path) { throw 'Skill points to another source.' }
        }
        Write-Output 'Shared per-skill views verified.'
        exit 0
    }
    if ((Resolve-Path -LiteralPath $item.Target).Path -ne (Resolve-Path -LiteralPath $canonical).Path) { throw 'Skill view points to a different source.' }
    Write-Output 'Shared skill view verified.'
    exit 0
}
New-Item -ItemType Directory -Path (Join-Path $repo '.claude') -Force | Out-Null
New-Item -ItemType Junction -Path $view -Target $canonical | Out-Null
Write-Output 'Shared skill view created.'
