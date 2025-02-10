# Quality Reports Documentation

This directory contains quality reports for river gauge data used in the Cornwall Flood Prediction project. These reports are crucial for understanding data reliability and identifying potential issues with gauge measurements.

## Directory Structure

```
reports/
├── quality/
│   ├── gauge_reports/          # Individual gauge quality reports
│   │   ├── boscastle/         
│   │   ├── bradworthy/
│   │   └── ...
│   └── aggregated/             # Overall quality analysis
```

## Quality Report Contents

Each gauge quality report includes:
- Data completeness analysis
- Quality flag distribution
- Temporal analysis of data quality
- Identification of suspicious measurements
- Visualization of quality patterns

## Metrics Used

The reports analyze several key metrics:
1. **Quality Flags**:
   - Good: High-quality measurements
   - Suspect: Potentially unreliable measurements
   - Missing: No data available
   - Error: Known measurement errors
   - Unchecked: Data pending quality review
   - Estimated: Values derived from estimation

2. **Data Quality Metrics**:
   - Completeness: Percentage of available data points
   - Consistency: Analysis of value distributions
   - Temporal patterns: Identification of quality trends

## Usage

1. Individual gauge reports can be found in `quality/gauge_reports/<gauge_name>/`
2. Each report is a Jupyter notebook that can be re-run to update the analysis
3. The `aggregated` directory contains overall quality assessments across all gauges

## Updating Reports

Reports should be updated:
- When new data is acquired
- If quality assessment methods change
- When investigating specific gauge issues
- Quarterly for regular monitoring

## Contributing

When creating new quality reports:
1. Use the template in `notebooks/examples/11_ea_data_quality_template.ipynb`
2. Follow the established color scheme and visualization standards
3. Place reports in appropriate gauge-specific directories
4. Update the aggregated analysis if necessary

## Contact

For questions about these reports or to report data quality issues, please open a GitHub issue.
