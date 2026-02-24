param(
    [string]$Owner = "ekkonwork",
    [string]$Repo = "qwen3-vl-autotagger-cli",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$fullRepo = "$Owner/$Repo"

Write-Host "Checking GitHub CLI auth..."
gh auth status | Out-Null

if (-not (git remote | Select-String -Pattern "^origin$" -Quiet)) {
    $visibilityArg = if ($Public) { "--public" } else { "--private" }
    Write-Host "Creating repository $fullRepo ($visibilityArg) and pushing..."
    gh repo create $fullRepo $visibilityArg --source . --remote origin --push --description "Standalone CLI for Qwen3-VL auto-tagging with optional XMP embedding."
} else {
    Write-Host "Remote 'origin' already exists. Skipping repo creation."
    Write-Host "Pushing current branch..."
    git push -u origin main
}

Write-Host "Applying repository topics..."
$topics = @(
    "qwen3-vl",
    "autotagging",
    "image-metadata",
    "xmp",
    "cli",
    "huggingface",
    "generative-ai",
    "stock-images"
)
foreach ($topic in $topics) {
    gh repo edit $fullRepo --add-topic $topic
}

Write-Host "Creating/updating issue labels..."
$labels = @(
    @{ name = "xmp"; color = "0E8A16"; description = "XMP metadata writing issues" },
    @{ name = "model-loading"; color = "B60205"; description = "Model loading, weights, or tokenizer failures" },
    @{ name = "dependencies"; color = "D4C5F9"; description = "Python/package installation or version conflicts" },
    @{ name = "performance"; color = "FBCA04"; description = "Speed, memory usage, or throughput improvements" },
    @{ name = "stock-quality"; color = "0052CC"; description = "Metadata quality for stock platform workflows" }
)
foreach ($label in $labels) {
    gh label create $label.name --repo $fullRepo --color $label.color --description $label.description --force
}

Write-Host "Done."
Write-Host "Repository URL:"
gh repo view $fullRepo --json url --jq ".url"

