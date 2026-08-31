"""A synthetic world, its document corpus, and multi-hop questions over it.

Why invent a world instead of using real facts: a question about the real
world can be answered from the base model's weights without ever calling the
search tool, so the reward would measure memorization and we would have no
way to tell the two apart. Every entity here is fictional, so the only route
to a correct answer is retrieval. That is the whole point of the category.

Why generate the questions from the entity graph instead of writing them by
hand: it guarantees every answer is actually supported by the corpus. A
hand-written multi-hop question whose second hop isn't in any document is an
unanswerable task that silently caps the achievable reward, and it is very
easy to write one by accident.

Each fact lives in exactly one document, and the chain is one-directional:
an expedition's document names its leader but not the instrument that leader
designed; the researcher's document names the instrument but not the
expedition. So a question spanning both cannot be answered from a single
retrieved document — which is what makes it genuinely multi-hop rather than
a lookup with extra words. `tests/test_qa_world.py` asserts that property
holds for every generated question rather than trusting the prose.
"""

import random
from dataclasses import dataclass

from .schema import Task

WORLD_SEED = 20260831

GIVEN_NAMES = (
    "Mira", "Tomas", "Ines", "Rafe", "Dalia", "Owen", "Petra", "Silas", "Noor",
    "Anders", "Yuki", "Cato", "Elin", "Bram", "Solveig", "Nadia", "Emeka", "Ivo",
)
SURNAMES = (
    "Halden", "Vance", "Oyelaran", "Brekke", "Cardew", "Fenn", "Aldritt",
    "Sorenson", "Quinlan", "Okonjo", "Verhoeven", "Lindqvist", "Abenov",
    "Rourke", "Delacroix", "Nayar", "Stroud", "Ferreira",
)

# Kept disjoint from SURNAMES so that a surname alias never matches two people.
CAPTAIN_NAMES = (
    "Greta Ashwell", "Milo Prentice", "Hanne Vikstrom", "Osric Bell",
    "Ada Marchetti", "Rune Halvorsen", "Isla Trant", "Kofi Danso",
    "Lorenz Kubik", "Sabine Wren-Fisher",
)

INSTRUMENTS = (
    "Kestrel Array", "Vantablade Sounder", "Orrery Magnetometer",
    "Lambent Spectrograph", "Talon Interferometer", "Cinder Profiler",
    "Marlow Anemograph", "Quillon Radiometer", "Vesper Tiltmeter",
    "Halcyon Photometer", "Bastion Seismograph", "Ferrule Bolometer",
    "Nimbus Hygrometer", "Kelter Densitometer", "Ardent Fluxgate",
    "Solent Coronagraph", "Palisade Scintillometer", "Mordant Nephelometer",
)
QUANTITIES = (
    "sub-surface brine salinity", "crustal magnetic declination",
    "aerosol backscatter", "ice-shelf basal melt rate",
    "atmospheric methane flux", "seafloor heat flow",
    "auroral electron precipitation", "snowpack liquid water content",
    "volcanic sulfur dioxide plumes", "geomagnetic micropulsations",
    "sea-ice thickness", "permafrost active-layer depth",
    "stratospheric ozone column", "tidal current shear",
    "cosmic ray muon flux", "soil radon emanation", "cloud droplet radius",
    "sediment turbidity",
)
SPECIALTIES = (
    "glaciology", "magnetospheric physics", "marine geochemistry",
    "atmospheric optics", "seismology", "palaeoclimatology", "benthic ecology",
    "cryospheric remote sensing", "volcanology", "physical oceanography",
    "aeronomy", "geodesy",
)

STATIONS = (
    "Anvil Point Station", "Corvid Bay Station", "Drossel Ridge Station",
    "Ember Sound Station", "Fenwold Station", "Grimsby Shoal Station",
    "Hollow Cape Station", "Ilsen Fjord Station",
)
REGIONS = (
    "the Sable Reach", "the Verrin Basin", "the Ostrand Shelf",
    "the Kalder Straits", "the Myrren Plateau", "the Tessine Barrens",
    "the Draymoor Coast", "the Wenlock Trench",
)

VESSELS = (
    "RV Lodestar", "RV Petrel", "RV Windrose", "RV Auger", "RV Marlin Fen",
    "RV Tessellate", "RV Gannet", "RV Northlight", "RV Kittiwake",
    "RV Sable Crane",
)
PORTS = (
    "Hallam Harbour", "Cove End", "Brackwater", "Ardmore Quay", "Sonnet Bay",
    "Tarn Harbour", "Wexford Landing", "Old Kiln Wharf", "Sperrin Docks",
    "Merrow Point",
)

EXPEDITIONS = (
    "Marrow Ridge Traverse", "Blackwater Shelf Survey", "Nine Fathom Expedition",
    "Cold Harbour Traverse", "Thistledown Survey", "Iron Gate Expedition",
    "Palewind Traverse", "Saltmarsh Survey", "Longshore Expedition",
    "Beacon Head Traverse", "Ashen Bay Survey", "Winterlight Expedition",
    "Redgrave Traverse", "Hollowmere Survey",
)

# Question subjects are partitioned so a held-out question is about an entity
# no training question ever asked about. The corpus itself is shared, as it
# has to be — it is the world both splits search.
HELDOUT_EXPEDITIONS = 4
HELDOUT_RESEARCHERS = 5
HELDOUT_STATIONS = 2


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Researcher:
    name: str
    surname: str
    specialty: str
    instrument: str
    joined: int
    doc_id: str


@dataclass(frozen=True)
class Instrument:
    name: str
    measures: str
    mass_kg: float
    in_service: int
    doc_id: str


@dataclass(frozen=True)
class Station:
    name: str
    region: str
    founded: int
    director: str
    berths: int
    doc_id: str


@dataclass(frozen=True)
class Vessel:
    name: str
    port: str
    captain: str
    length_m: int
    doc_id: str


@dataclass(frozen=True)
class Expedition:
    name: str
    leader: str
    station: str
    vessel: str
    year: int
    duration_weeks: int
    doc_id: str


@dataclass(frozen=True)
class World:
    researchers: tuple[Researcher, ...]
    instruments: tuple[Instrument, ...]
    stations: tuple[Station, ...]
    vessels: tuple[Vessel, ...]
    expeditions: tuple[Expedition, ...]
    documents: tuple[Document, ...]

    def researcher(self, name: str) -> Researcher:
        return next(r for r in self.researchers if r.name == name)

    def instrument(self, name: str) -> Instrument:
        return next(i for i in self.instruments if i.name == name)

    def station(self, name: str) -> Station:
        return next(s for s in self.stations if s.name == name)

    def vessel(self, name: str) -> Vessel:
        return next(v for v in self.vessels if v.name == name)

    def document(self, doc_id: str) -> Document:
        return next(d for d in self.documents if d.doc_id == doc_id)


def build_world(seed: int = WORLD_SEED) -> World:
    """Build the corpus. Deterministic, so eval numbers stay reproducible."""
    rng = random.Random(seed)

    instruments = tuple(
        Instrument(
            name=name,
            measures=QUANTITIES[i],
            mass_kg=round(rng.uniform(2.5, 180.0), 1),
            in_service=rng.randint(2026, 2038),
            doc_id=f"ins-{i:02d}",
        )
        for i, name in enumerate(INSTRUMENTS)
    )

    researchers = tuple(
        Researcher(
            name=f"{GIVEN_NAMES[i]} {SURNAMES[i]}",
            surname=SURNAMES[i],
            specialty=rng.choice(SPECIALTIES),
            instrument=instruments[i].name,
            joined=rng.randint(2024, 2039),
            doc_id=f"res-{i:02d}",
        )
        for i in range(len(SURNAMES))
    )

    directors = rng.sample(researchers, len(STATIONS))
    stations = tuple(
        Station(
            name=name,
            region=REGIONS[i],
            founded=rng.randint(1998, 2032),
            director=directors[i].name,
            berths=rng.randrange(12, 90),
            doc_id=f"sta-{i:02d}",
        )
        for i, name in enumerate(STATIONS)
    )

    vessels = tuple(
        Vessel(
            name=name,
            port=PORTS[i],
            captain=CAPTAIN_NAMES[i],
            length_m=rng.randrange(38, 122),
            doc_id=f"ves-{i:02d}",
        )
        for i, name in enumerate(VESSELS)
    )

    expeditions = tuple(
        Expedition(
            name=name,
            leader=rng.choice(researchers).name,
            station=rng.choice(stations).name,
            vessel=rng.choice(vessels).name,
            year=rng.randint(2031, 2044),
            duration_weeks=rng.randrange(3, 26),
            doc_id=f"exp-{i:02d}",
        )
        for i, name in enumerate(EXPEDITIONS)
    )

    documents = (
        tuple(_instrument_doc(i) for i in instruments)
        + tuple(_researcher_doc(r) for r in researchers)
        + tuple(_station_doc(s) for s in stations)
        + tuple(_vessel_doc(v) for v in vessels)
        + tuple(_expedition_doc(e) for e in expeditions)
    )

    return World(researchers, instruments, stations, vessels, expeditions, documents)


# --------------------------------------------------------------------------
# document rendering
#
# Filler sentences carry only facts that belong to the document's own entity.
# Anything that leaked a second entity's fact would collapse a two-hop
# question into a one-hop lookup without the question changing at all.
# --------------------------------------------------------------------------

def _instrument_doc(ins: Instrument) -> Document:
    return Document(
        doc_id=ins.doc_id,
        title=ins.name,
        text=(
            f"The {ins.name} is a field instrument that measures {ins.measures}. "
            f"It has a dry mass of {ins.mass_kg} kg and entered routine service in "
            f"{ins.in_service}. The instrument is calibrated before each deployment "
            f"and logs continuously to an internal store."
        ),
    )


def _researcher_doc(res: Researcher) -> Document:
    return Document(
        doc_id=res.doc_id,
        title=res.name,
        text=(
            f"{res.name} is a researcher working in {res.specialty}. "
            f"{res.surname} joined the institute in {res.joined} and designed the "
            f"{res.instrument}, which is still in use. {res.surname} supervises "
            f"graduate students and reviews field proposals each season."
        ),
    )


def _station_doc(sta: Station) -> Document:
    return Document(
        doc_id=sta.doc_id,
        title=sta.name,
        text=(
            f"{sta.name} stands on {sta.region} and was founded in {sta.founded}. "
            f"The station is directed by {sta.director} and has {sta.berths} berths. "
            f"It stays open through the winter and supports visiting field parties."
        ),
    )


def _vessel_doc(ves: Vessel) -> Document:
    return Document(
        doc_id=ves.doc_id,
        title=ves.name,
        text=(
            f"{ves.name} is a {ves.length_m}-metre research vessel home-ported at "
            f"{ves.port}. She is captained by {ves.captain} and carries a crew of "
            f"eighteen. The vessel is ice-strengthened and refits every third year."
        ),
    )


def _expedition_doc(exp: Expedition) -> Document:
    return Document(
        doc_id=exp.doc_id,
        title=f"The {exp.name}",
        text=(
            f"The {exp.name} took place in {exp.year} and ran for "
            f"{exp.duration_weeks} weeks. It was led by {exp.leader}. The party "
            f"was based at {exp.station} and travelled aboard {exp.vessel}. "
            f"Field notes and raw logs from the {exp.name} are held in the "
            f"institute archive."
        ),
    )


# --------------------------------------------------------------------------
# question generation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    template: str
    subject: str
    text: str
    answers: tuple[str, ...]
    supporting_docs: tuple[str, ...]
    hops: int


def _person_aliases(full_name: str) -> tuple[str, ...]:
    return (full_name, full_name.split()[-1])


def _expedition_questions(world: World, exp: Expedition) -> list[Question]:
    leader = world.researcher(exp.leader)
    station = world.station(exp.station)
    vessel = world.vessel(exp.vessel)
    instrument = world.instrument(leader.instrument)

    return [
        Question(
            template="leader_instrument",
            subject=exp.name,
            text=f"Which instrument was designed by the researcher who led the {exp.name}?",
            answers=(instrument.name,),
            supporting_docs=(exp.doc_id, leader.doc_id),
            hops=2,
        ),
        Question(
            template="leader_specialty",
            subject=exp.name,
            text=f"What field does the researcher who led the {exp.name} work in?",
            answers=(leader.specialty,),
            supporting_docs=(exp.doc_id, leader.doc_id),
            hops=2,
        ),
        Question(
            template="station_region",
            subject=exp.name,
            text=f"In which region is the station that the {exp.name} was based at?",
            answers=(station.region, station.region.removeprefix("the ")),
            supporting_docs=(exp.doc_id, station.doc_id),
            hops=2,
        ),
        Question(
            template="vessel_port",
            subject=exp.name,
            text=f"Where is the home port of the vessel that carried the {exp.name}?",
            answers=(vessel.port,),
            supporting_docs=(exp.doc_id, vessel.doc_id),
            hops=2,
        ),
        Question(
            template="vessel_captain",
            subject=exp.name,
            text=f"Who captains the vessel that carried the {exp.name}?",
            answers=_person_aliases(vessel.captain),
            supporting_docs=(exp.doc_id, vessel.doc_id),
            hops=2,
        ),
        Question(
            template="leader_instrument_measures",
            subject=exp.name,
            text=(
                f"What does the instrument designed by the researcher who led the "
                f"{exp.name} measure?"
            ),
            answers=(instrument.measures,),
            supporting_docs=(exp.doc_id, leader.doc_id, instrument.doc_id),
            hops=3,
        ),
    ]


def _researcher_questions(world: World, res: Researcher) -> list[Question]:
    instrument = world.instrument(res.instrument)
    return [
        Question(
            template="researcher_instrument_measures",
            subject=res.name,
            text=f"What does the instrument designed by {res.name} measure?",
            answers=(instrument.measures,),
            supporting_docs=(res.doc_id, instrument.doc_id),
            hops=2,
        )
    ]


def _station_questions(world: World, sta: Station) -> list[Question]:
    director = world.researcher(sta.director)
    return [
        Question(
            template="director_instrument",
            subject=sta.name,
            text=f"Which instrument was designed by the director of {sta.name}?",
            answers=(director.instrument,),
            supporting_docs=(sta.doc_id, director.doc_id),
            hops=2,
        ),
        Question(
            template="director_specialty",
            subject=sta.name,
            text=f"What field does the director of {sta.name} work in?",
            answers=(director.specialty,),
            supporting_docs=(sta.doc_id, director.doc_id),
            hops=2,
        ),
    ]


def build_questions(world: World, split: str) -> list[Question]:
    """Every question for `split`, over that split's share of the subjects."""
    if split == "train":
        expeditions = world.expeditions[:-HELDOUT_EXPEDITIONS]
        researchers = world.researchers[:-HELDOUT_RESEARCHERS]
        stations = world.stations[:-HELDOUT_STATIONS]
    else:
        expeditions = world.expeditions[-HELDOUT_EXPEDITIONS:]
        researchers = world.researchers[-HELDOUT_RESEARCHERS:]
        stations = world.stations[-HELDOUT_STATIONS:]

    questions: list[Question] = []
    for exp in expeditions:
        questions.extend(_expedition_questions(world, exp))
    for res in researchers:
        questions.extend(_researcher_questions(world, res))
    for sta in stations:
        questions.extend(_station_questions(world, sta))
    return questions


def generate_qa_tasks(split: str, world: World | None = None) -> list[Task]:
    world = world if world is not None else build_world()
    return [
        Task(
            task_id=f"qa-{split}-{i:04d}",
            category="qa",
            prompt=question.text,
            ground_truth={
                "answers": list(question.answers),
                "supporting_docs": list(question.supporting_docs),
            },
            split=split,
            metadata={
                "template": question.template,
                "subject": question.subject,
                "hops": question.hops,
            },
        )
        for i, question in enumerate(build_questions(world, split))
    ]
