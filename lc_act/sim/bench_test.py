import tyro

from lc_act.sim import bench as _bench


def test_first_init_offsets_the_init_range_of_every_task():
    args = tyro.cli(
        _bench.Args, args=["--ckpt", "x.pt", "--tasks", "libero_10/8", "--episodes", "3", "--first-init", "20"]
    )
    jobs = _bench.pick_jobs(args)
    assert jobs == [_bench.Job("libero_10", 8, 20), _bench.Job("libero_10", 8, 21), _bench.Job("libero_10", 8, 22)]


def test_default_covers_all_suites_from_init_zero():
    jobs = _bench.pick_jobs(tyro.cli(_bench.Args, args=["--ckpt", "x.pt", "--episodes", "2"]))
    assert len(jobs) == 4 * 10 * 2
    assert {job.init for job in jobs} == {0, 1}


def test_several_checkpoints_follow_one_flag():
    args = tyro.cli(_bench.Args, args=["--ckpt", "a.pt", "b.pt", "--suites", "libero_10", "libero_goal"])
    assert [path.name for path in args.ckpt] == ["a.pt", "b.pt"]
    assert args.suites == ("libero_10", "libero_goal")


def test_deal_keeps_each_task_on_one_worker():
    jobs = [_bench.Job("libero_10", task, init) for task in range(3) for init in range(2)]
    queues = _bench._deal(jobs, 2)
    assert [{job.task_id for job in queue} for queue in queues] == [{0, 2}, {1}]


def test_summarize_averages_suites():
    results = {
        _bench.Job("libero_10", 0, 0): (True, 10),
        _bench.Job("libero_10", 0, 1): (False, 520),
        _bench.Job("libero_goal", 1, 0): (True, 30),
    }
    summary = _bench.summarize(results)
    assert summary["suites"] == {"libero_10": 0.5, "libero_goal": 1.0}
    assert summary["average"] == 0.75
    assert summary["tasks"]["libero_10/0"] == 0.5
