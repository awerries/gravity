use rand::Rng;
use rand::distributions::Uniform;
use bevy::{prelude::*, sprite::MaterialMesh2dBundle, window::PresentMode};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

// http://arborjs.org/docs/barnes-hut
const WINDOW_SIZE: f32 = 1200.;
const THETA_THRESHOLD: f32 = 0.5;
const PARTICLE_MASS: f32 = 1e7;
const GRAVITY: f32 = 6.6743e-11; // m^3 / (kg s^2)
const DSCALE: f32 = 1000.; // distance scaling
const SIM_SPEED: f32 = 10.0;
const NUM_PARTICLES: u32 = 20000;

const INIT_VEL_FACTOR: f32 = 4.; // arbitrary scalar on num_particles to limit initial velocity
const VEL_VARIATION: f32 = 0.3;
const MIN_DIST: f32 = 0.5;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        .register_type::<Velocity>()
        .register_type::<Force>()
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
        .add_system(update)
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

    for _ in 0..NUM_PARTICLES {
        let (r, theta, pos) = random_pos(window.width());
        spawn_particle(&mut commands, &mut meshes, &mut materials, pos, random_vel(r, theta));
    }
}

fn random_pos(width: f32) -> (f32, f32, Vec3)
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0., 1.);
    let r = (width / 2.0 * rng.sample(dist)).sqrt()*4.;
    let theta = rng.sample(dist) * 2. * 3.1415927;
    (r, theta, Vec3::new(r*theta.cos(), r*theta.sin(), 0.))
}

fn random_vel(r: f32, theta: f32) -> Vec2
{
    let mut rng = rand::thread_rng();
    let v_orbit = (GRAVITY * PARTICLE_MASS * (NUM_PARTICLES as f32) / INIT_VEL_FACTOR / r).sqrt();

    let v = v_orbit + v_orbit * rng.gen_range(-VEL_VARIATION..VEL_VARIATION);

    Vec2::new(v*theta.sin(), -v*theta.cos())
}


fn spawn_particle(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    pos: Vec3,
    vel: Vec2,
) {
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(1.).into()).into(),
            material: materials.add(ColorMaterial::from(Color::WHITE)),
            transform: Transform::from_translation(pos),
            ..default()
        },
        Velocity(vel),
        Force(Vec2::new(0., 0.))
    ));
}

fn update(
    windows: Res<Windows>,
    par_commands: ParallelCommands,
    mut particle_query: Query<(&mut Transform, Entity, &mut Velocity, &mut Force)>
) {
    let window = windows.get_primary().unwrap();

    // each cycle, we build the quad-tree out of the particles
    let mut tree = BHTree::new(Quadrant {length: window.width(), corner: Vec2::new(-window.width()/2., -window.width()/2.)});
    for (transform, particle, _, _) in &particle_query {
        tree.insert(Body::new(transform.translation.truncate(), particle))
    }

    // update the position of each particle
    particle_query.par_for_each_mut(NUM_PARTICLES as usize / 64, |(mut transform, particle, mut velocity, mut force)| {
        // TODO avoid recreating Body in each loop
        **force = tree.get_force(&Body::new(transform.translation.truncate(), particle));
        let accel = **force / PARTICLE_MASS;
        let t = SIM_SPEED;
        transform.translation.x += velocity.x*t + 0.5 * accel.x * t * t;
        transform.translation.y += velocity.y*t + 0.5 * accel.y * t * t;
        velocity.x += accel.x * t;
        velocity.y += accel.y * t;

        // despawn particles that go out of bounds (The other option is to crash! :P or simulate far outside of bounds so they can come back)
        if (transform.translation.x.abs() >= window.width() / 2.0) || (transform.translation.y.abs() >= window.width() / 2.0) {
            par_commands.command_scope(|mut commands| {
                commands.entity(particle).despawn();
            });
        }
    });
}

#[derive(Component, Deref, DerefMut, Reflect)]
struct Velocity(Vec2);

#[derive(Component, Deref, DerefMut, Reflect)]
struct Force(Vec2);

#[derive(Default)]
struct Quadrant {
    corner: Vec2,
    length: f32
}

impl Quadrant {
    /// return true if this Quadrant contains (x,y)
    fn contains(&self, x: f32, y: f32) -> bool {
        (x >= self.corner.x) && (x < self.corner.x + self.length) && (y >= self.corner.y) && (y < self.corner.y + self.length)
    }
    fn nw(&self) -> Self {
        let half = self.length / 2.0;
        Quadrant {corner: Vec2::new(self.corner.x, self.corner.y + half), length: half}
    }
    fn ne(&self) -> Self {
        let half = self.length / 2.0;
        Quadrant {corner: Vec2::new(self.corner.x + half, self.corner.y + half), length: half}
    }
    fn sw(&self) -> Self {
        let half = self.length / 2.0;
        Quadrant {corner: Vec2::new(self.corner.x, self.corner.y), length: half}
    }
    fn se(&self) -> Self {
        let half = self.length / 2.0;
        Quadrant {corner: Vec2::new(self.corner.x + half, self.corner.y), length: half}
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
    fn new(pos: Vec2, particle: Entity) -> Self {
        Body {
            mass: PARTICLE_MASS,
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
        let dist = b.pos.distance(self.pos);
        // protect against tiny floating point values, otherwise we get insane acceleration without collision detection
        if dist > MIN_DIST {
            // F_vec = -G * M * r_vec / |r|^3
            - GRAVITY * self.mass * b.mass * (self.pos - b.pos) / dist.powi(3)
        } else {
            Vec2::new(0., 0.)
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
            // We have a body, but does it have an particle or is it internal (and is therefore a group)?
            if current_body.particle.is_some() {
                // this body is a particle, so we have two particles, so need to split this region
                self.subtree = Some(SubQuadrants::new(&self.quad));
                // copy all the fields, as we're moving the particle down a layer
                let copy = current_body.clone();
                // clear the actual particle, as this is now an internal node
                current_body.particle = None;
                // add the new particle to the now-internal node mass and COM
                current_body.add(&b);
                // add both particles as external nodes
                self.insert_to_quadrant(copy);
                self.insert_to_quadrant(b);
            } else {
                current_body.add(&b);
                self.insert_to_quadrant(b);
            }
        } else {
            // current node has no body, add it here as particle/external
            self.body = Some(b);
        }
    }

    fn insert_to_quadrant(&mut self, b: Body) {
        // this is an internal node, we must have a subtree
        let subtree = self.subtree.as_mut().unwrap();
        match b {
            b if b.within(&subtree.nw.quad) => subtree.nw.insert(b),
            b if b.within(&subtree.ne.quad) => subtree.ne.insert(b),
            b if b.within(&subtree.sw.quad) => subtree.sw.insert(b),
            b if b.within(&subtree.se.quad) => subtree.se.insert(b),
            b => panic!("position {}, {} was not in any quadrant?", b.pos.x, b.pos.y)
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
                if self.quad.length / dist < THETA_THRESHOLD {
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
            nw: Box::new(BHTree::new(q.nw())),
            ne: Box::new(BHTree::new(q.ne())),
            sw: Box::new(BHTree::new(q.sw())),
            se: Box::new(BHTree::new(q.se())),
        }
    }

    fn get_force(&self, b: &Body) -> Vec2 {
        self.nw.get_force(b) + self.ne.get_force(b) + self.sw.get_force(b) + self.se.get_force(b)
    }
}