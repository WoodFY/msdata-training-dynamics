import os
import sys
import argparse
import subprocess
from pathlib import Path


def generate_batch_script(script_path, input_file, output_dir):
    base_name = os.path.basename(input_file)
    file_name, _ = os.path.splitext(base_name)
    output_file_path = os.path.join(output_dir, f"{file_name}_peaks.csv")
    script_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <batch mzmine_version="4.7.8">
        <batchstep method="io.github.mzmine.modules.io.import_rawdata_all.AllSpectralDataImportModule" parameter_version="1">
            <parameter name="File names"><file>{input_file}</file></parameter>
            <parameter name="Advanced import" selected="true">
                <parameter name="Scan filters" selected="true"><parameter name="Polarity">POSITIVE</parameter></parameter>
            </parameter>
        </batchstep>
        
        <batchstep method="io.github.mzmine.modules.dataprocessing.featdet_massdetection.MassDetectionModule" parameter_version="1">
            <parameter name="Raw data files" type="BATCH_LAST_FILES"/>
            <parameter name="Scan filters" selected="true">
                <parameter name="MS level filter" selected="MS1, level = 1">1</parameter>
            </parameter>
            <parameter name="Mass detector" selected_item="Auto">
                <module name="Auto">
                    <parameter name="Noise level">100000.0</parameter>
                </module>
            </parameter>
        </batchstep>
        
        <batchstep method="io.github.mzmine.modules.dataprocessing.featdet_adapchromatogrambuilder.ModularADAPChromatogramBuilderModule" parameter_version="1">
            <parameter name="Raw data files" type="BATCH_LAST_FILES"/>
            <parameter name="Scan filters" selected="true">
                <parameter name="MS level filter" selected="MS1, level = 1">1</parameter>
            </parameter>
            <parameter name="Minimum consecutive scans">5</parameter>
            <parameter name="Minimum absolute height">100000.0</parameter>
            <parameter name="m/z tolerance (scan-to-scan)">
                <absolutetolerance>0.001</absolutetolerance>
                <ppmtolerance>5.0</ppmtolerance>
            </parameter>
            <parameter name="Suffix">chromatograms</parameter>
        </batchstep>
        
        <batchstep method="io.github.mzmine.modules.io.export_features_csv_legacy.LegacyCSVExportModule" parameter_version="1">
            <parameter name="Feature lists" type="BATCH_LAST_FEATURELISTS"/>
            <parameter name="Filename">
                <current_file>{output_file_path}</current_file>
                <last_file>{output_file_path}</last_file>
            </parameter>
            <parameter name="Field separator">,</parameter>
            <parameter name="Export common elements"><item>Export row m/z</item><item>Export row retention time</item></parameter>
            <parameter name="Export data file elements"><item>Peak area</item><item>Peak height</item></parameter>
            <parameter name="Export quantitation results and other information">false</parameter>
            <parameter name="Identification separator">;</parameter>
            <parameter name="Filter rows">ALL</parameter>
        </batchstep>
    </batch>
    """
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)


def generate_alignment_batch_script(script_path, input_files, output_dir):
    file_tags = "\n".join([f"            <file>{file_path}</file>" for file_path in input_files])
    output_file_path = os.path.join(output_dir, "aligned_feature_list.csv")
    script_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <batch mzmine_version="4.7.8">
        <batchstep method="io.github.mzmine.modules.io.import_rawdata_all.AllSpectralDataImportModule" parameter_version="1">
            <parameter name="File names">
    {file_tags}
            </parameter>
            <parameter name="Advanced import" selected="true">
                <parameter name="Scan filters" selected="true">
                    <parameter name="Polarity">POSITIVE</parameter>
                </parameter>
            </parameter>
        </batchstep>

        <batchstep method="io.github.mzmine.modules.dataprocessing.featdet_massdetection.MassDetectionModule" parameter_version="1">
            <parameter name="Raw data files" type="BATCH_LAST_FILES"/>
            <parameter name="Scan filters" selected="true">
                <parameter name="MS level filter" selected="MS1, level = 1">1</parameter>
            </parameter>
            <parameter name="Mass detector" selected_item="Auto">
                <module name="Auto">
                    <parameter name="Noise level">100000.0</parameter>
                </module>
            </parameter>
        </batchstep>

        <batchstep method="io.github.mzmine.modules.dataprocessing.featdet_adapchromatogrambuilder.ModularADAPChromatogramBuilderModule" parameter_version="1">
            <parameter name="Raw data files" type="BATCH_LAST_FILES"/>
            <parameter name="Scan filters" selected="true">
                <parameter name="MS level filter" selected="MS1, level = 1">1</parameter>
            </parameter>
            <parameter name="Minimum consecutive scans">5</parameter>
            <parameter name="Minimum intensity for consecutive scans">100000.0</parameter>
            <parameter name="Minimum absolute height">100000.0</parameter>
            <parameter name="m/z tolerance (scan-to-scan)">
                <absolutetolerance>0.001</absolutetolerance>
                <ppmtolerance>5.0</ppmtolerance>
            </parameter>
            <parameter name="Suffix">chromatograms</parameter>
        </batchstep>

        <batchstep method="io.github.mzmine.modules.dataprocessing.align_join.JoinAlignerModule" parameter_version="1">
            <parameter name="Feature lists" type="BATCH_LAST_FEATURELISTS"/>
            <parameter name="Feature list name">Aligned feature list</parameter>
            <parameter name="m/z tolerance (sample-to-sample)">
                <absolutetolerance>0.001</absolutetolerance>
                <ppmtolerance>5.0</ppmtolerance>
            </parameter>
            <parameter name="Weight for m/z">2.0</parameter>
            <parameter name="Retention time tolerance" unit="MINUTES">0.1</parameter>
            <parameter name="Weight for RT">1.0</parameter>
            <parameter name="Require same charge state">false</parameter>
            <parameter name="Require same ID">false</parameter>
        </batchstep>

        <batchstep method="io.github.mzmine.modules.io.export_features_csv_legacy.LegacyCSVExportModule" parameter_version="1">
            <parameter name="Feature lists" type="BATCH_LAST_FEATURELISTS"/>
            <parameter name="Filename">
                <current_file>{output_file_path}</current_file>
                <last_file>{output_file_path}</last_file>
            </parameter>
            <parameter name="Field separator">,</parameter>
            <parameter name="Export common elements">
                <item>Export row m/z</item>
                <item>Export row retention time</item>
            </parameter>
            <parameter name="Export data file elements">
                <item>Peak area</item>
                <item>Peak height</item>
            </parameter>
            <parameter name="Export quantitation results and other information">false</parameter>
            <parameter name="Identification separator">;</parameter>
            <parameter name="Filter rows">ALL</parameter>
        </batchstep>
    </batch>
    """
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)


def main():
    parser = argparse.ArgumentParser(description='使用MZmine4.3.0进行峰提取并导出CSV')
    parser.add_argument('--mzmine_path', default=r"D:\Softwares\MZmine\mzmine.exe", help='MZmine executable file path.')
    parser.add_argument('--mzmine_user', default=r"C:\Users\Wood IVVI\.mzmine\users\woodfy.mzuser", help='MZmine user configuration file path.')
    parser.add_argument('--input_dir', required=True, help='Input directory path, containing multiple .mzML files')
    parser.add_argument('--output_dir', default='mzmine_output', help='Output directory path')
    parser.add_argument('--align', action='store_true', help='Whether to perform alignment across multiple files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ms_file_paths = list(Path(args.input_dir).glob('*.mzML'))

    if not ms_file_paths:
        print(f"Error: No .mzML files found in {args.input_dir}.")
        sys.exit(1)

    if args.align:
        if len(ms_file_paths) < 2:
            print(f"Error: at least 2 .mzML files are needed to perform alignment.")
            sys.exit(1)

        output_csv = os.path.join(args.output_dir, "aligned_feature_list.csv")
        if os.path.exists(output_csv):
            print(f"{output_csv} already exists, skipping.")
            sys.exit(1)

        batch_script_path = os.path.join(args.output_dir, 'script', f'align_batch_script.mzb')
        os.makedirs(os.path.dirname(batch_script_path), exist_ok=True)
        generate_alignment_batch_script(batch_script_path, [str(p) for p in ms_file_paths], args.output_dir)

        mzmine_cmd = [
            args.mzmine_path,
            '--user', args.mzmine_user,
            '--batch', batch_script_path
        ]
        try:
            subprocess.run(mzmine_cmd, capture_output=True, text=True, check=True, shell=True)
            print("MZmine alignment process completed!")
        except subprocess.CalledProcessError as e:
            print(f"MZmine alignment process failed! {e}")
            sys.exit(1)
    else:
        for ms_file_path in ms_file_paths:
            output_csv = os.path.join(args.output_dir, f"{ms_file_path.stem}_peaks.csv")

            if os.path.exists(output_csv):
                print(f"{output_csv} already exists, skipping file: {ms_file_path.name}")
                continue

            batch_script_path = os.path.join(args.output_dir, 'script', f'{ms_file_path.stem}_batch_script.mzb')
            generate_batch_script(batch_script_path, str(ms_file_path), args.output_dir)

            mzmine_cmd = [
                args.mzmine_path,
                '--user', args.mzmine_user,
                '--batch', batch_script_path
            ]
            try:
                subprocess.run(mzmine_cmd, capture_output=True, text=True, check=True, shell=True)
                print("MZmine process completed!")
            except subprocess.CalledProcessError as e:
                print(f"MZmine process failed! {e}")
                sys.exit(1)

    print("Processing completed!")

if __name__ == "__main__":
    main()