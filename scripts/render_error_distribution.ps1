param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$payload = Get-Content -LiteralPath $InputPath -Raw -Encoding UTF8 | ConvertFrom-Json
$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
try {
    $chart.Width = 1400
    $chart.Height = 600
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = 'All'
    $chart.TextAntiAliasingQuality = 'High'
    $title = $chart.Titles.Add([string]$payload.title)
    $title.Font = New-Object System.Drawing.Font('Segoe UI', 13)
    $title.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(0, 0, 100, 12)
    $cdf = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('cdf')
    $cdf.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(2, 22, 47, 67)
    $cdf.AxisX.IsLogarithmic = $true
    $cdf.AxisX.LogarithmBase = 10
    $cdf.AxisX.LabelStyle.Format = '0.#E+0'
    $cdf.AxisX.Title = 'Per-sample mean relative error (%) [log axis]'
    $cdf.AxisY.Title = 'Cumulative samples (%)'
    $cdf.AxisY.Minimum = 0
    $cdf.AxisY.Maximum = 100
    $cdf.AxisY.Interval = 25
    $chart.ChartAreas.Add($cdf)
    $hist = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea('histogram')
    $hist.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(51, 22, 47, 67)
    $hist.AxisX.Title = 'Per-sample mean relative error (%)'
    $hist.AxisX.Minimum = 0
    $hist.AxisY.Title = 'Density'
    $hist.AxisY.Minimum = 0
    $hist.AxisX.LabelStyle.Format = '0.###'
    $hist.AxisY.LabelStyle.Format = '0.##E+0'
    $chart.ChartAreas.Add($hist)
    foreach ($area in @($cdf, $hist)) {
        $area.BackColor = [System.Drawing.Color]::White
        $area.AxisX.TitleFont = New-Object System.Drawing.Font('Segoe UI', 10)
        $area.AxisY.TitleFont = New-Object System.Drawing.Font('Segoe UI', 10)
        $area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 9)
        $area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Segoe UI', 9)
        $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(225, 230, 235)
        $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::FromArgb(225, 230, 235)
    }
    $leftTitle = $chart.Titles.Add('All defined samples')
    $leftTitle.Font = New-Object System.Drawing.Font('Segoe UI', 11)
    $leftTitle.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(5, 15, 41, 6)
    $rightTitle = $chart.Titles.Add('95th-percentile view (plot only)')
    $rightTitle.Font = New-Object System.Drawing.Font('Segoe UI', 11)
    $rightTitle.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(55, 15, 41, 6)
    $line = New-Object System.Windows.Forms.DataVisualization.Charting.Series('CDF')
    $line.ChartArea = 'cdf'
    $line.ChartType = 'StepLine'
    $line.Color = [System.Drawing.Color]::FromArgb(35, 80, 135)
    $line.BorderWidth = 3
    foreach ($point in $payload.cdf) {
        [void]$line.Points.AddXY([double]$point[0], ([double]$point[1] * 100.0))
    }
    $chart.Series.Add($line)
    $bars = New-Object System.Windows.Forms.DataVisualization.Charting.Series('density')
    $bars.ChartArea = 'histogram'
    $bars.ChartType = 'Column'
    $bars.Color = [System.Drawing.Color]::FromArgb(70, 110, 160)
    $bars['PointWidth'] = '0.98'
    foreach ($point in $payload.histogram) {
        [void]$bars.Points.AddXY([double]$point[0], [double]$point[1])
    }
    $chart.Series.Add($bars)
    # Explicit labels avoid Chart's automatic suppression of fractional Y ticks.
    $cdf.AxisY.IsLabelAutoFit = $false
    $cdf.AxisY.LabelStyle.Enabled = $true
    $cdf.AxisY.LabelStyle.ForeColor = [System.Drawing.Color]::Black
    foreach ($value in @(0.0, 25.0, 50.0, 75.0, 100.0)) {
        $label = New-Object System.Windows.Forms.DataVisualization.Charting.CustomLabel
        $label.FromPosition = $value - 12.5
        $label.ToPosition = $value + 12.5
        $label.Text = $value.ToString('0.##', [System.Globalization.CultureInfo]::InvariantCulture)
        [void]$cdf.AxisY.CustomLabels.Add($label)
    }
    $peak = ($payload.histogram | ForEach-Object { [double]$_[1] } | Measure-Object -Maximum).Maximum
    $hist.AxisY.Maximum = [Math]::Max([double]$peak * 1.15, 1e-12)
    $step = $hist.AxisY.Maximum / 4.0
    $hist.AxisY.Interval = $step
    $hist.AxisY.IsLabelAutoFit = $false
    $hist.AxisY.LabelStyle.Enabled = $true
    $hist.AxisY.LabelStyle.ForeColor = [System.Drawing.Color]::Black
    for ($tick = 0; $tick -le 4; $tick++) {
        $value = $tick * $step
        $label = New-Object System.Windows.Forms.DataVisualization.Charting.CustomLabel
        $label.FromPosition = $value - $step / 2.0
        $label.ToPosition = $value + $step / 2.0
        $label.Text = $value.ToString('0.##E+0', [System.Globalization.CultureInfo]::InvariantCulture)
        [void]$hist.AxisY.CustomLabels.Add($label)
    }
    $footer = $chart.Titles.Add([string]$payload.footer)
    $footer.Font = New-Object System.Drawing.Font('Segoe UI', 9)
    $footer.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(2, 94, 96, 5)
    $chart.SaveImage($OutputPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
}
finally {
    $chart.Dispose()
}
