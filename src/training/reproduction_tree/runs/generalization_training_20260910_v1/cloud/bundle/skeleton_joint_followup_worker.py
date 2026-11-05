"""Use the unchanged sensor training loop for two controlled joint encoders."""
from pathlib import Path
import argparse, json, sys, time, traceback


def main():
    a = argparse.ArgumentParser()
    for name in ['root', 'job', 'protocol']:a.add_argument('--'+name, required=True)
    a.add_argument('--deadline', type=float, required=True)
    args = a.parse_args();root = Path(args.root);sys.path.insert(0, str(root/'bundle'))
    from dual_host_runtime.io import atomic, read
    from dual_host_runtime.health import Guard
    from research_runtime.common import configure, sha
    import research_runtime.skeleton_joint_followup_train as train
    from skeleton_joint_model import SensorTemporal
    from skeleton_joint_base import SensorTemporal as FlatSensor
    configure();p = read(args.protocol);p.update(_sha=sha(args.protocol), _bundle=str(root/'bundle'))
    spec = p['jobs'][args.job];out = root/'output'/args.job;out.mkdir(parents=True, exist_ok=True)
    guard = Guard(out, min(args.deadline,p['budget']['stage_deadline_unix']), root/'TELEMETRY.json')
    try:
        assert not set(spec['train_users']) & set(spec['predict_users'])
        assert not set(spec['train_users']) & set(p['folds'][str(spec['fold'])])
        p['_seed'] = spec['seed']
        for name, expected in p['code_sha256'].items():assert sha(root/'bundle'/name) == expected, name
        assert sha(root/'bundle/data/measured_sensors.npz') == p['measured_data_sha256']
        p['calibration'] = dict(encoder_fit_users=spec['train_users'], calibration_users=spec['predict_users'], assessment_users=[])
        train.SensorConstructor = FlatSensor if spec['arm']=='flat' else lambda: SensorTemporal(graph=spec['graph'])
        guard.check(force=True, phase='INITIALIZING_JOINT_CONTROL', job=args.job)
        train.train_sensor(p, out, guard)
        atomic(out/'INTERVENTION.json', dict(arm=spec['arm'], graph=spec['graph'], protocol_sha256=p['_sha'],
               train_users=spec['train_users'], predict_users=spec['predict_users'], fixed_epochs=60,
               test_access=False, skeletal_joint_names_assumed=False))
    except BaseException as e:
        atomic(out/'ERROR.json', dict(error=repr(e), traceback=traceback.format_exc(), at_unix=time.time()));raise


if __name__ == '__main__':main()
