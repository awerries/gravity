use rand::Rng;
use rand::distributions::Uniform;
use bevy::{
    prelude::*,
    sprite::MaterialMesh2dBundle,
    window::PresentMode,
    time::FixedTimestep
};
use glam::DVec2;
use random_color;
use random_color::RandomColor;

const WINDOW_SIZE: f32 = 1200.;
const OUT_OF_BOUNDS: f64 = 2.0; // track particles this far out of bounds

// http://arborjs.org/docs/barnes-hut
const THETA_THRESHOLD: f64 = 0.9;

const GRAVITY: f64 = 6.6743e-10; // m^3 / (kg s^2)
const DSCALE: f64 = 1e18; // distance scaling w.r.t. pixels. average distance between stars in milky way is 5 light years, or 4.73e16 meters
const SIM_STEP: f64 = 1e15; // time of each sim step, in seconds. 1e12 seconds is 31.7k years. takes 230million years for sun to get around milky way.

const FPS: f64 = 30.0;
const TIME_STEP: f64 = 1.0/FPS; // how often bevy will attempt to run the sim, in seconds

const NUM_PARTICLES: u32 = 10000;
const AVG_PARTICLE_MASS: f64 = 1e31; // mass of sun is around 2e30 kg
const PARTICLE_MAG_VARIATION: f64 = 1.1;

const VEL_VARIATION: f64 = 0.05;
const GALAXY_WIDTH_SCALE: f64 = 0.20;
const GALAXY_HEIGHT_SCALE: f64 = 2.;

// Minimum radius to guard against gravity singularity
const MIN_R: f64 = DSCALE;
const MIN_R2: f64 = MIN_R*MIN_R;

// Min grid size to protect against floating point division errors
const MIN_QUADRANT_LENGTH: f64 = 0.000001 * DSCALE;


#[derive(Component)]
struct Pose {
    r: DVec2,
    v: DVec2
}

#[derive(Component)]
struct Mass(f64);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum AppState {
    Running,
    Paused,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                title: "gravity_sim".to_string(),
                width: WINDOW_SIZE,
                height: WINDOW_SIZE,
                present_mode: PresentMode::AutoVsync,
                ..default()
            },
            ..default()
        }))
        .add_state(AppState::Paused)
        .add_startup_system(setup)
        .add_system_set(
            SystemSet::new()
            .with_run_criteria(FixedTimestep::step(TIME_STEP as f64))
            .with_system(update)
        )
        //.add_system_set(SystemSet::on_update(AppState::Running).with_system(update))
        .add_system(detect_pause)
        .add_system(bevy::window::close_on_esc)
        .run();
}

fn detect_pause(mut app_state: ResMut<State<AppState>>, mut keys: ResMut<Input<KeyCode>>) {
    if keys.just_pressed(KeyCode::Space) {
        let current = app_state.current().clone();
        app_state
            .set(match current {
                AppState::Running => AppState::Paused,
                AppState::Paused => AppState::Running,
            })
            .unwrap();
        // ^ this can fail if we are already in the target state
        // or if another state change is already queued;
        keys.reset(KeyCode::Space);
    }
}

fn setup(
    windows: Res<Windows>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let window = windows.get_primary().unwrap();
    commands.spawn(Camera2dBundle::default());

    let w = window.width() as f64 * DSCALE;
    spawn_galaxy(
        &mut commands,
        &mut meshes,
        &mut materials,
        NUM_PARTICLES / 2,
        w * GALAXY_WIDTH_SCALE,
        DVec2::new(w / 8.0, w / 8.0),
        DVec2::new(0., -0.001)
    );
    spawn_galaxy(
        &mut commands,
        &mut meshes,
        &mut materials,
        NUM_PARTICLES / 2,
        w * GALAXY_WIDTH_SCALE,
        DVec2::new(-w / 8.0, -w / 8.0),
        DVec2::new(0., 0.001)
    );
}

fn spawn_galaxy(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<ColorMaterial>>,
    num_particles: u32,
    diameter: f64,
    gpos: DVec2,
    gvel: DVec2
) {
    let gmass = (num_particles as f64) * AVG_PARTICLE_MASS;
    for _ in 0..num_particles {
        let (r, theta, pos) = random_circle_pos(diameter);
        let vel = random_orbital_circle_vel(r, theta, diameter*GALAXY_HEIGHT_SCALE, gmass);
        spawn_particle(&mut commands, &mut meshes, &mut materials, pos + gpos, vel + gvel);
    }
}

fn random_circle_pos(diameter: f64) -> (f64, f64, DVec2)
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0., 1.);
    let r = diameter / 2.0 * rng.sample(dist);
    let theta = rng.sample(dist) * 2. * 3.1415927;
    (r, theta, DVec2::new(r*theta.cos(), r*theta.sin()))
}

fn random_orbital_circle_vel(r: f64, theta: f64, scale_height: f64, total_mass: f64) -> DVec2
{
    let mut rng = rand::thread_rng();
    let grav = GRAVITY * total_mass / (r * r + scale_height * scale_height).sqrt();
    let v = (2.0 * grav).sqrt();
    DVec2::new(
        v*theta.sin() + v * rng.gen_range(-VEL_VARIATION..VEL_VARIATION),
        -v*theta.cos() + v * rng.gen_range(-VEL_VARIATION..VEL_VARIATION)
    )
}

fn spawn_particle(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    pos: DVec2,
    vel: DVec2,
) {
    let color = RandomColor::new().hue(random_color::Color::Blue).to_rgb_array();
    let mut rng = rand::thread_rng();
    let mdist = Uniform::new(AVG_PARTICLE_MASS / PARTICLE_MAG_VARIATION, AVG_PARTICLE_MASS * PARTICLE_MAG_VARIATION);
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(1.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::rgb_u8(color[0], color[1], color[2]))),
            transform: Transform::from_translation(Vec3::new(pos.x as f32, pos.y as f32, 0.) / DSCALE as f32),
            ..default()
        },
        Pose {r: pos, v: vel},
        Mass(rng.sample(mdist))
    ));
}

fn update(
    windows: Res<Windows>,
    par_commands: ParallelCommands,
    mut particle_query: Query<(&mut Transform, Entity, &mut Pose, &Mass)>
) {
    let window = windows.get_primary().unwrap();

    // each cycle, we build the quad-tree out of the particles
    let mut tree = BHTree::new(Quadrant::new(OUT_OF_BOUNDS * window.width() as f64 * DSCALE));
    for (_, particle, pose, mass) in &particle_query {
        tree.insert(particle, Body::new(mass, pose.r))
    }

    // update the position of each particle
    particle_query.par_for_each_mut(512, |(mut transform, particle, mut pose, mass)| {
        let Mass(mass_f) = *mass;
        // TODO avoid recreating Body in each loop
        let accel = tree.get_force(&particle, &Body::new(mass, pose.r)) / mass_f;
        let t = SIM_STEP;
        let v_i = pose.v;
        let dv = accel * t;
        pose.r += (v_i + 0.5 * dv) * t;
        pose.v += dv;

        // set actual rendering position
        transform.translation.x = (pose.r.x / DSCALE) as f32;
        transform.translation.y = (pose.r.y / DSCALE) as f32;

        // despawn particles that go out of bounds (The other option is to crash! :P or simulate far outside of bounds so they can come back)
        if (transform.translation.x.abs() >= window.width() / 2.0 * OUT_OF_BOUNDS as f32 - 1.0)
        || (transform.translation.y.abs() >= window.width() / 2.0 * OUT_OF_BOUNDS as f32 - 1.0)
        {
            par_commands.command_scope(|mut commands| {
                commands.entity(particle).despawn();
            });
        }
    });
}


enum Corner {NW, NE, SW, SE}

#[derive(Default, Debug)]
struct Quadrant {
    center: DVec2,
    len: f64
}

impl Quadrant {
    fn new(length: f64) -> Self {
        Quadrant {
            len: length,
            center: DVec2::new(0., 0.)
        }
    }
    /// return true if this Quadrant contains (x,y)
    fn contains(&self, x: f64, y: f64) -> bool {
        let hl = self.len / 2.0;
        (x >= self.center.x - hl) && (x < self.center.x + hl) && (y >= self.center.y - hl) && (y < self.center.y + hl)
    }
    
    fn subquad(&self, corner: Corner) -> Self {
        let hl = self.len / 2.0;
        let ql = hl / 2.0;
        match corner {
            Corner::NW => Quadrant {center: DVec2::new(self.center.x - ql, self.center.y + ql), len: hl},
            Corner::NE => Quadrant {center: DVec2::new(self.center.x + ql, self.center.y + ql), len: hl},
            Corner::SW => Quadrant {center: DVec2::new(self.center.x - ql, self.center.y - ql), len: hl},
            Corner::SE => Quadrant {center: DVec2::new(self.center.x + ql, self.center.y - ql), len: hl}
        }
    }
}


enum NodeItem {
    Internal(SubQuadrants),
    Leaf(Entity)
}

struct Node {
    body: Body,
    item: NodeItem
}

impl Node {
    fn new(body: Body, item: NodeItem) -> Self {
        Node {body, item}
    }
}

#[derive(Default)]
struct BHTree
{
    quad: Quadrant,
    node: Option<Node>
}

impl BHTree
{
    fn new(quad: Quadrant) -> Self {
        BHTree {quad, ..default()}
    }

    fn insert(&mut self, particle: Entity, body: Body) {
        if let Some(current_node) = &mut self.node {
            match &mut current_node.item {
                NodeItem::Internal(subquad) => {
                    current_node.body.add(&body);
                    subquad.insert_to_quadrant(particle, body);
                },
                NodeItem::Leaf(node_particle) => {
                    if self.quad.len > MIN_QUADRANT_LENGTH {
                        // we only have 1 particle per region, so we split and generate subtrees
                        let mut subquad = SubQuadrants::new(&self.quad);
                        subquad.insert_to_quadrant(node_particle.clone(), current_node.body);
                        subquad.insert_to_quadrant(particle, body);
                        current_node.item = NodeItem::Internal(subquad);
                    }
                    // implied else: if we've already got too small of a grid, we still add the mass for a cheap estimate
                    current_node.body.add(&body);
                }
            }
        } else {
            // current node has no body, add it here as particle/external
            self.node = Some(Node::new(body, NodeItem::Leaf(particle)));
        }
    }

    fn get_force(&self, p: &Entity, b: &Body) -> DVec2 {
        if let Some(current_node) = &self.node {
            match &current_node.item {
                NodeItem::Internal(subquad) => {
                    let dist = current_node.body.pos.distance(b.pos);
                    if self.quad.len / dist < THETA_THRESHOLD {
                        // treat node as a single body
                        b.force(&current_node.body)
                    } else {
                        // traverse the tree, returning the total force
                        subquad.get_force(p, b)
                    }
                },
                NodeItem::Leaf(node_particle) => {
                    if node_particle.index() != p.index() {
                        b.force(&current_node.body)
                    } else {
                        // index was the same, this is the same particle
                        DVec2::new(0., 0.)
                    }
                }
            }
        } else {
            // there's no body at self, so there's no force
            DVec2::new(0., 0.)
        }
    }
}


struct SubQuadrants {
    nw: Box<BHTree>,
    ne: Box<BHTree>,
    sw: Box<BHTree>,
    se: Box<BHTree>
}

impl SubQuadrants {
    fn new(q: &Quadrant) -> Self {
        SubQuadrants {
            nw: Box::new(BHTree::new(q.subquad(Corner::NW))),
            ne: Box::new(BHTree::new(q.subquad(Corner::NE))),
            sw: Box::new(BHTree::new(q.subquad(Corner::SW))),
            se: Box::new(BHTree::new(q.subquad(Corner::SE))),
        }
    }

    fn get_force(&self, p: &Entity, b: &Body) -> DVec2 {
        self.nw.get_force(p, b) + self.ne.get_force(p, b) + self.sw.get_force(p, b) + self.se.get_force(p, b)
    }

    fn insert_to_quadrant(&mut self, p: Entity, b: Body) {
        // this is an internal node, we must have a subtree
        match b {
            b if b.within(&self.nw.quad) => self.nw.insert(p, b),
            b if b.within(&self.ne.quad) => self.ne.insert(p, b),
            b if b.within(&self.sw.quad) => self.sw.insert(p, b),
            b if b.within(&self.se.quad) => self.se.insert(p, b),
            b => panic!("position {}, {} was not in any quadrant?\n {:#?}, {:#?}, {:#?}, {:#?}", b.pos.x, b.pos.y, self.nw.quad, self.ne.quad, self.sw.quad, self.se.quad)
        }
    }
}


#[derive(Default, Clone, Copy)]
struct Body {
    mass: f64,
    pos: DVec2,
}

impl Body {
    fn new(mass: &Mass, pos: DVec2) -> Self {
        let Mass(mass) = *mass;
        Body {mass, pos}
    }

    fn within(&self, q: &Quadrant) -> bool {
        q.contains(self.pos.x, self.pos.y)
    }

    fn add(&mut self, b: &Body) {
        // get total mass, but don't set overall mass until after we update COM
        let total_mass = self.mass + b.mass;
        // update center of mass
        self.pos.x = (self.pos.x * self.mass + b.pos.x * b.mass) / total_mass;
        self.pos.y = (self.pos.y * self.mass + b.pos.y * b.mass) / total_mass;
        self.mass = total_mass;
    }

    /// compute force on self from Body
    fn force(&self, b: &Body) -> DVec2 {
        let dist2 = b.pos.distance_squared(self.pos);
        // protect against tiny floating point values, otherwise we get insane acceleration without collision detection
        // Barnes. "Gravitational softening as a smoothing operation"
        // https://home.ifa.hawaii.edu/users/barnes/research/smoothing/soft.pdf
        // F_vec = -G * M1 * M2 * (r2-r1) / (|r2-r1|^2 + eps^2)^3/2
        - GRAVITY * self.mass * b.mass * (self.pos - b.pos) / (dist2 + MIN_R2).powf(3./2.)
    }
}