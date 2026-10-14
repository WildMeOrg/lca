"""The GZCD selection, pinned against the run the paper reports.

Two sets are built from the release and both matter. The clustered set is the
dataset row of Table 1. The embedding set is wider, and because the classifier
threshold is fitted on every embedding in the file rather than on what is
clustered, getting it wrong moves the threshold and nothing downstream matches.

What can go wrong is the rule that recognises a census region: the export merged
each body-plus-region pair into one row, so an unmerged region is a standalone
row whose `bbox` is the region itself. Mistaking a body detection for one of
those, or missing one, changes the set silently. Those cases are pinned here.
The end-to-end counts need the release, so that check is download-gated.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import reproduce_gzcd

#: What a run clusters, and what gets embedded.
ANNOTATIONS, INDIVIDUALS, TO_EMBED = 3764, 435, 4501

#: The annotations the identification stage dropped, which no public field marks.
EXCLUSIONS = 43


def annotation(**overrides):
    base = {'image_uuid': 'image', 'annot_uuid': 'annotation',
            'bbox': [0, 0, 100, 100], 'region_bbox': [10, 10, 50, 50],
            'category_id': 0, 'viewpoint': 'right', 'individual_id': 'animal',
            'annotations_census': True, 'manual': False, 'CA_score': 0.99}
    return {**base, **overrides}


def regions(*annotations):
    return reproduce_gzcd.census_regions({'annotations': list(annotations)})


def test_a_box_is_read_from_either_a_list_or_its_text():
    assert reproduce_gzcd.box([1.4, 2.6, 3, 4]) == [1, 3, 3, 4]
    assert reproduce_gzcd.box('[1, 2, 3, 4]') == [1, 2, 3, 4]


def test_a_rows_own_region_box_is_a_census_region():
    found = regions(annotation())
    assert [bbox for _, bbox in found] == [[10, 10, 50, 50]]


def test_the_box_cropped_is_the_region_not_the_detection():
    found = regions(annotation())
    assert found[0][1] != found[0][0]['bbox']


def test_an_unmerged_region_is_found_inside_its_own_animals_detection():
    """Its bbox is the region; the detection is a separate row for that animal."""
    detection = annotation(annot_uuid='detection', region_bbox=None)
    region = annotation(annot_uuid='region', region_bbox=None,
                        bbox=[10, 10, 50, 50])
    found = regions(detection, region)
    assert {a['annot_uuid'] for a, _ in found} == {'region'}


def test_an_animals_only_box_is_taken_as_a_region():
    """Its detection is not in the release, so containment cannot be checked."""
    found = regions(annotation(region_bbox=None))
    assert [bbox for _, bbox in found] == [[0, 0, 100, 100]]


def test_a_hand_added_row_without_a_region_is_not_one():
    """Those are later additions, and none of them is a census region."""
    assert regions(annotation(region_bbox=None, manual=True)) == []


def test_a_detection_that_encloses_the_others_is_not_a_region():
    big = annotation(annot_uuid='big', region_bbox=None, bbox=[0, 0, 100, 100])
    small = annotation(annot_uuid='small', region_bbox=None, bbox=[10, 10, 50, 50])
    found = regions(big, small)
    assert 'big' not in {a['annot_uuid'] for a, _ in found}


def test_species_and_viewpoint_are_filtered():
    assert regions(annotation(category_id=1)) == []
    assert regions(annotation(viewpoint='left')) == []
    assert regions(annotation(individual_id=None)) == []


def test_a_row_is_keyed_by_the_identifier_the_pipeline_reads():
    """The pipeline reads `uuid`; the release calls that field `annot_uuid`."""
    rows = reproduce_gzcd.rows_for(regions(annotation()))
    assert rows[0]['uuid'] == 'annotation'
    assert 'annot_uuid' not in rows[0]


def test_one_row_per_image_and_box():
    rows = reproduce_gzcd.rows_for(regions(annotation(), annotation()))
    assert len(rows) == 1


def test_the_exclusions_ship_with_the_script():
    assert len(reproduce_gzcd.read_exclusions()) == EXCLUSIONS


def test_the_clustered_set_drops_the_listed_exclusions():
    excluded = sorted(reproduce_gzcd.read_exclusions())[0]
    metadata = {'annotations': [annotation(annot_uuid=excluded),
                                annotation(annot_uuid='kept', image_uuid='other')],
                'categories': []}
    clustering, embedding = reproduce_gzcd.select(metadata)
    assert {row['uuid'] for row in clustering} == {'kept'}
    assert {row['uuid'] for row in embedding} == {excluded, 'kept'}


@pytest.mark.skipif('--run-download' not in sys.argv,
                    reason='needs the release; run with --run-download')
def test_the_release_gives_the_reported_dataset(tmp_path):
    metadata = reproduce_gzcd.download_metadata(tmp_path)
    clustering, embedding = reproduce_gzcd.select(metadata)
    assert len(clustering) == ANNOTATIONS
    assert len({row['individual_id'] for row in clustering}) == INDIVIDUALS
    assert len(embedding) == TO_EMBED
    # the threshold is fitted on the embedding file, so it must contain every
    # node, or the join drops annotations and a smaller dataset is clustered
    assert {row['uuid'] for row in clustering} <= {row['uuid'] for row in embedding}
