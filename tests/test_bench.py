from lc_act.bench import Job, parse_args, pick_jobs


def test_first_init_offsets_the_init_range_of_every_task():
    jobs = pick_jobs(parse_args(["--ckpt", "x.pt", "--tasks", "libero_10/8", "--episodes", "3", "--first-init", "20"]))
    assert jobs == [Job("libero_10", 8, 20), Job("libero_10", 8, 21), Job("libero_10", 8, 22)]


def test_default_covers_all_suites_from_init_zero():
    jobs = pick_jobs(parse_args(["--ckpt", "x.pt", "--episodes", "2"]))
    assert len(jobs) == 4 * 10 * 2
    assert {job.init for job in jobs} == {0, 1}
