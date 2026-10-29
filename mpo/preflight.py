"""Verify that the Ray cluster can reach retrieval and read judge configuration."""
import argparse
import os
import urllib.request


def check_node(rag_url):
    from verl.utils.reward_score.bc_p_llm_judge import get_judge_config
    get_judge_config()  # Validate credentials locally; never return or print them.
    with urllib.request.urlopen(rag_url.rstrip('/') + '/health', timeout=30) as response:
        if response.status != 200:
            raise RuntimeError('Retrieval health check failed')
    return True


def main():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, required=True)
    args = parser.parse_args()
    ray.init(address='auto', logging_level='ERROR', runtime_env={'env_vars': {
        'BCP_JUDGE_ENV_FILE': os.getenv('BCP_JUDGE_ENV_FILE', ''),
    }})
    nodes = [node for node in ray.nodes() if node.get('Alive')]
    if len(nodes) != args.nodes:
        raise RuntimeError(f'Expected {args.nodes} live Ray nodes, found {len(nodes)}')
    remote = ray.remote(num_cpus=0)(check_node)
    tasks = [remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node['NodeID'], soft=False)).remote(os.environ['RAG_URL']) for node in nodes]
    ray.get(tasks)
    print(f'Preflight passed on {len(nodes)} nodes')
    ray.shutdown()


if __name__ == '__main__':
    main()
