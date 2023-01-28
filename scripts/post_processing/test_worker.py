import daisy
import logging
import time
import pymongo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_worker():

    db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017"
    db_name="zebrafish_postprocessing"

    # open block done DB
    mongo_client = pymongo.MongoClient(db_host)
    db = mongo_client[db_name]
    blocks_extracted = db[f"16_test_blocks_completed"]

    rag_provider = daisy.persistence.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        edges_collection=f"16_test_edges",
        nodes_collection=f"16_test_nodes",
        meta_collection=f"16_test_meta",
    )

    client = daisy.Client()

    while True:
        logger.info("getting block")
        with client.acquire_block() as block:

            logger.info(f"got block {block}")
            time.sleep(0.5)

            if block is None:
                break

            start = time.time()
            document = {
                "num_cpus": 5,
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            blocks_extracted.insert_one(document)
            logger.info(f"releasing block: {block}")


if __name__ == "__main__":

    test_worker()
