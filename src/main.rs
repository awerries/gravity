use rand::Rng;
use rand::distributions::Uniform;
use bevy::{
    prelude::*,
    sprite::MaterialMesh2dBundle,
    window::PresentMode,
    time::FixedTimestep
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use random_color;
use random_color::RandomColor;

const WINDOW_SIZE: f32 = 1200.;

// http://arborjs.org/docs/barnes-hut
const THETA_THRESHOLD: f32 = 1.5;

const GRAVITY: f32 = 6.6743e-11; // m^3 / (kg s^2)
const DSCALE: f32 = 1_000.; // distance scaling w.r.t. meters
const SIM_STEP: f32 = 1.0; // time of each sim step, in seconds

const FPS: f32 = 30.0;
const TIME_STEP: f32 = 1.0/FPS; // how often bevy will attempt to run the sim, in seconds

const NUM_PARTICLES: u32 = 20000;
const AVG_PARTICLE_MASS: f32 = 1e13;
const PARTICLE_MAG_VARIATION: f32 = 1.1;

const VEL_VARIATION: f32 = 0.2;
const GALAXY_WIDTH_SCALE: f32 = 0.8;
const GALAXY_HEIGHT_SCALE: f32 = 1.0;

// Minimum radius to guard against gravity singularity
const MIN_R: f32 = 0.01 * DSCALE;
const MIN_R2: f32 = MIN_R*MIN_R;

// Min grid size to protect against floating point division errors
const QUADRANT_SCALE: f32 = 2.0;
const MIN_QUADRANT_LENGTH: f32 = MIN_R * QUADRANT_SCALE;


#[derive(Component)]
struct Pose {
    r: Vec2,
    v: Vec2
}

#[derive(Component)]
struct Mass(f32);


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
        //.add_plugin(WorldInspectorPlugin)
        .add_startup_system(setup)
        .add_system_set(
            SystemSet::new()
            .with_run_criteria(FixedTimestep::step(TIME_STEP as f64))
            .with_system(update)
        )
        .add_system(bevy::window::close_on_esc)
        .run();
}

fn setup(
    windows: Res<Windows>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let window = windows.get_primary().unwrap();
    commands.spawn(Camera2dBundle::default());

    spawn_galaxy(&mut commands, &mut meshes, &mut materials, window.width() * GALAXY_WIDTH_SCALE);
    //spawn_grid(&mut commands, &mut meshes, &mut materials, window.width()/ 1.1);
}


fn spawn_particle(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    pos: Vec3,
    vel: Vec2,
) {
    let color = RandomColor::new().hue(random_color::Color::Blue).to_rgb_array();
    let mut rng = rand::thread_rng();
    let mdist = Uniform::new(AVG_PARTICLE_MASS / PARTICLE_MAG_VARIATION, AVG_PARTICLE_MASS * PARTICLE_MAG_VARIATION);
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(1.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::rgb_u8(color[0], color[1], color[2]))),
            transform: Transform::from_translation(pos),
            ..default()
        },
        Pose {r: pos.truncate() * DSCALE, v: vel},
        Mass(rng.sample(mdist))
    ));
}

fn spawn_galaxy(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<ColorMaterial>>,
    diameter: f32
) {
    for _ in 0..NUM_PARTICLES {
        let (r, theta, pos) = random_circle_pos(diameter);
        let vel = random_orbital_circle_vel(r * DSCALE, theta, DSCALE*diameter*GALAXY_HEIGHT_SCALE);
        //let vel = Vec2::new(0., 0.);
        spawn_particle(&mut commands, &mut meshes, &mut materials, pos, vel);
    }
}

fn random_circle_pos(diameter: f32) -> (f32, f32, Vec3)
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0., 1.);
    let r = diameter / 2.0 * rng.sample(dist);
    let theta = rng.sample(dist) * 2. * 3.1415927;
    (r, theta, Vec3::new(r*theta.cos(), r*theta.sin(), 0.))
}

fn random_orbital_circle_vel(r: f32, theta: f32, scale_height: f32) -> Vec2
{
    let mut rng = rand::thread_rng();
    let grav = GRAVITY * AVG_PARTICLE_MASS * (NUM_PARTICLES as f32) / (r * r + scale_height * scale_height).sqrt();
    let v_orbit = (2.0 * grav).sqrt();
    let v = v_orbit + v_orbit * rng.gen_range(-VEL_VARIATION..VEL_VARIATION);
    Vec2::new(v*theta.sin(), -v*theta.cos())
}

fn update(
    windows: Res<Windows>,
    par_commands: ParallelCommands,
    mut particle_query: Query<(&mut Transform, Entity, &mut Pose, &Mass)>
) {
    let window = windows.get_primary().unwrap();

    // each cycle, we build the quad-tree out of the particles
    let mut tree = BHTree::new(Quadrant::new(window.width() * DSCALE));
    for (transform, particle, _, mass) in &particle_query {
        tree.insert(Body::new(mass, transform.translation.truncate() * DSCALE, particle))
    }

    // update the position of each particle
    particle_query.par_for_each_mut(NUM_PARTICLES as usize / 1024, |(mut transform, particle, mut pose, mass)| {
        let Mass(mass_f) = *mass;
        // TODO avoid recreating Body in each loop
        let accel = tree.get_force(&Body::new(mass, pose.r, particle)) / mass_f;
        let t = SIM_STEP;
        let v_i = pose.v;
        let dv = accel * t;
        pose.r += (v_i + 0.5 * dv) * t;
        pose.v += dv;

        // set actual rendering position
        transform.translation.x = pose.r.x / DSCALE;
        transform.translation.y = pose.r.y / DSCALE;

        // despawn particles that go out of bounds (The other option is to crash! :P or simulate far outside of bounds so they can come back)
        if (transform.translation.x.abs() >= window.width() / 2.0 - 1.0) || (transform.translation.y.abs() >= window.width() / 2.0 - 1.0) {
            par_commands.command_scope(|mut commands| {
                commands.entity(particle).despawn();
            });
        }
    });
}


enum Corner {NW, NE, SW, SE}

#[derive(Default, Debug)]
struct Quadrant {
    center: Vec2,
    len: f32 // half-length
}

impl Quadrant {
    fn new(length: f32) -> Self {
        Quadrant {
            len: length,
            center: Vec2::new(0., 0.)
        }
    }
    /// return true if this Quadrant contains (x,y)
    fn contains(&self, x: f32, y: f32) -> bool {
        let hl = self.len / 2.0;
        let e = f32::EPSILON; // ensure that we don't have floating point grid gaps
        (x >= self.center.x - hl) && (x < self.center.x + hl + e) && (y >= self.center.y - hl ) && (y < self.center.y + hl + e)
    }
    
    fn subquad(&self, corner: Corner) -> Self {
        let hl = self.len / 2.0;
        let ql = hl / 2.0;
        match corner {
            Corner::NW => Quadrant {center: Vec2::new(self.center.x - ql, self.center.y + ql), len: hl},
            Corner::NE => Quadrant {center: Vec2::new(self.center.x + ql, self.center.y + ql), len: hl},
            Corner::SW => Quadrant {center: Vec2::new(self.center.x - ql, self.center.y - ql), len: hl},
            Corner::SE => Quadrant {center: Vec2::new(self.center.x + ql, self.center.y - ql), len: hl}
        }
    }
}


#[derive(Default)]
struct BHTree
{
    // if there is no body, the BHTree is empty
    // if the body has no particle, this BHTree will have a subtree.
    // TODO: make this not be such an asinine struct relationship
    body: Option<Body>,
    quad: Quadrant,
    subtree: Option<SubQuadrants>
}

impl BHTree
{
    fn new(quad: Quadrant) -> Self {
        // is it weird to impl new just for this?
        BHTree {
            quad,
            ..default()
        }
    }

    fn insert(&mut self, b: Body) {
        if let Some(current_body) = &mut self.body {
            // We have a body, but does it have an particle or is it a group?
            if current_body.particle.is_some() {
                if self.quad.len > MIN_QUADRANT_LENGTH {
                    // this body is a particle, and we only have 1 particle per region if over the min length, so we split and generate subtrees
                    self.subtree = Some(SubQuadrants::new(&self.quad));
                    // copy all the fields, as we're moving the particle down a layer
                    let copy = current_body.clone();
                    // clear the actual particle, as this is now an internal node
                    current_body.particle = None;
                    // add the new particle to the now-internal node mass and COM
                    current_body.add(&b);

                    // add both particles as external nodes
                    let subtree = self.subtree.as_mut().unwrap();
                    subtree.insert_to_quadrant(copy);
                    subtree.insert_to_quadrant(b);
                } else {
                    // we've already got too small of a grid, just add the mass for a cheap estimate
                    current_body.add(&b);
                }
            } else {
                current_body.add(&b);
                self.subtree.as_mut().unwrap().insert_to_quadrant(b);
            }
        } else {
            // current node has no body, add it here as particle/external
            self.body = Some(b);
        }
    }



    fn get_force(&self, b: &Body) -> Vec2 {     
        if let Some(node_body) = &self.body {
            if let Some(node_particle) = &node_body.particle {
                if node_particle.index() != b.particle.unwrap().index() {
                    b.force(node_body)
                } else {
                    // index was the same, this is the same particle
                    Vec2::new(0., 0.)
                }
            } else {
                // current node does not have a particle, but does have a body, and therefore must be an internal node
                let dist = node_body.pos.distance(b.pos);
                if self.quad.len / dist < THETA_THRESHOLD {
                    // treat node as a single body
                    b.force(node_body)
                } else {
                    // traverse the tree, returning the total force
                    self.subtree.as_ref().unwrap().get_force(b)
                }
            }
        } else {
            // there's no body at self, so there's no force
            Vec2::new(0., 0.)
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

    fn get_force(&self, b: &Body) -> Vec2 {
        self.nw.get_force(b) + self.ne.get_force(b) + self.sw.get_force(b) + self.se.get_force(b)
    }

    fn insert_to_quadrant(&mut self, b: Body) {
        // this is an internal node, we must have a subtree
        match b {
            b if b.within(&self.nw.quad) => self.nw.insert(b),
            b if b.within(&self.ne.quad) => self.ne.insert(b),
            b if b.within(&self.sw.quad) => self.sw.insert(b),
            b if b.within(&self.se.quad) => self.se.insert(b),
            b => panic!("position {}, {} was not in any quadrant?\n {:#?}, {:#?}, {:#?}, {:#?}", b.pos.x, b.pos.y, self.nw.quad, self.ne.quad, self.sw.quad, self.se.quad)
        }
    }
}


#[derive(Default, Clone)]
struct Body {
    mass: f32,
    pos: Vec2,
    // if body has a particle entity, it's an external particle.
    // if not, it's a collection of particles.
    particle: Option<Entity>
}

impl Body {
    fn new(mass: &Mass, pos: Vec2, particle: Entity) -> Self {
        let Mass(mass) = mass;
        Body {
            mass: *mass,
            pos,
            particle: Some(particle)
        }
    }

    fn within(&self, q: &Quadrant) -> bool {
        q.contains(self.pos.x, self.pos.y)
    }

    fn add(&mut self, b: &Body) {
        // get total mass, but don't set overall mass until after we update COM
        let total_mass = self.mass + b.mass;
        // update center of mass
        self.pos.x = (self.pos.x*self.mass + b.pos.x*b.mass) / total_mass;
        self.pos.y = (self.pos.y*self.mass + b.pos.y*b.mass) / total_mass;
        self.mass = total_mass;
    }

    /// compute force on self from Body
    fn force(&self, b: &Body) -> Vec2 {
        let dist2 = b.pos.distance_squared(self.pos);
        // protect against tiny floating point values, otherwise we get insane acceleration without collision detection
        // Barnes. "Gravitational softening as a smoothing operation"
        // https://home.ifa.hawaii.edu/users/barnes/research/smoothing/soft.pdf
        // F_vec = -G * M * (r2-r1) / (|r2-r1|^2 + eps^2)^3/2
        - GRAVITY * self.mass * b.mass * (self.pos - b.pos) / (dist2 + MIN_R2).powf(3./2.)
    }
}