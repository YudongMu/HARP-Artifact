"""Repair Decord 0.6's stale internal cp36 tag to its published py3 wheel tag.

The published artifact is decord-0.6.0-py3-none-manylinux2010_x86_64.whl.
It uses ctypes/libdecord.so; native decoding is verified before this repair.
No Python code or native library is modified.
"""
import base64,csv,hashlib,importlib.metadata,io,json,pathlib


def repair_decord_metadata(root=None):
    root=pathlib.Path(root) if root is not None else pathlib.Path(importlib.metadata.distribution('decord')._path)
    if root.name!='decord-0.6.0.dist-info':raise ValueError('Repair is specific to Decord 0.6.0')
    wheel=root/'WHEEL';old=wheel.read_text()
    tags=[line for line in old.splitlines() if line.startswith('Tag:')]
    expected='Tag: cp36-cp36m-manylinux2010_x86_64';replacement='Tag: py3-none-manylinux2010_x86_64'
    if tags==[replacement]:return {'changed':False,'wheel_tag':replacement}
    if tags!=[expected]:raise ValueError('Unknown wheel ABI/platform; refusing relabel')
    rows=list(csv.reader((root/'RECORD').read_text().splitlines()))
    target=root.name+'/WHEEL'
    if not any(row[0]==target for row in rows):raise ValueError('Missing WHEEL integrity record')
    new=old.replace(expected,replacement).encode()
    for row in rows:
        if row[0]==target:
            row[1]='sha256='+base64.urlsafe_b64encode(hashlib.sha256(new).digest()).decode().rstrip('=');row[2]=str(len(new))
    output=io.StringIO(newline='');csv.writer(output,lineterminator='\n').writerows(rows)
    wheel.write_bytes(new);(root/'RECORD').write_text(output.getvalue())
    return {'changed':True,'old_tag':expected,'new_tag':replacement,'old_wheel_sha256':hashlib.sha256(old.encode()).hexdigest(),
            'new_wheel_sha256':hashlib.sha256(new).hexdigest(),'native_library_modified':False,
            'published_artifact':'decord-0.6.0-py3-none-manylinux2010_x86_64.whl'}


if __name__=='__main__':
    import decord  # Fail before relabeling if the native library cannot load.
    report=repair_decord_metadata()
    pathlib.Path('/opt/harp/env/decord-metadata-repair.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
