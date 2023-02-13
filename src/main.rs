use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use bevy_inspector_egui::quick::WorldInspectorPlugin;

// http://arborjs.org/docs/barnes-hut
const WINDOW_SIZE: f32 = 1000.;
const THETA_THRESHOLD: f32 = 0.5;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0., 0., 0.)))
        .add_plugins(DefaultPlugins)
        .add_plugin(WorldInspectorPlugin)
        .add_startup_system(setup)
        .add_system(update)
        .add_system(bevy::window::close_on_esc)
        .run();
}

fn setup(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());

    spawn_particle(commands, meshes, materials);
}

fn spawn_particle(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(5.).into()).into(),
            material: materials.add(ColorMaterial::from(Color::WHITE)),
            transform: Transform::from_translation(Vec3::new(-100., 0., 0.)),
            ..default()
        },
        Velocity(Vec2::new(0., 0.))
    ));
}

fn update(
    particle_query: Query<(&mut Transform, Entity)>
) {
    let mut tree = BHTree::new(Quadrant {length: WINDOW_SIZE, ..default()});
    for (transform, particle) in &particle_query {
        tree.insert(Body{mass: 1., pos: transform.translation.truncate(), particle: Some(particle)})
    }
}

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Default)]
struct Quadrant {
    corner: Vec2,
    length: f32
}

impl Quadrant {
    /// return true if this Quadrant contains (x,y)
    fn contains(&self, x: f32, y: f32) -> bool {
        (x > self.corner.x) && (x < self.corner.x + self.length) && (y > self.corner.y) && (y < self.corner.y + self.length)
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
    particle: Option<Entity>
}

impl Body {
    fn within(&self, q: &Quadrant) -> bool {
        q.contains(self.pos.x, self.pos.y)
    }

    fn add(&mut self, b: &Body) {
        let total_mass = self.mass + b.mass;
        self.pos.x = (self.pos.x*self.mass + b.pos.x*b.mass) / total_mass;
        self.pos.y = (self.pos.y*self.mass + b.pos.y*b.mass) / total_mass;
        self.mass = total_mass;
    }
}

#[derive(Default)]
struct BHTree
{
    body: Option<Body>,
    quad: Quadrant,
    subtree: Option<SubQuadrants>
}

impl BHTree
{
    fn new(quad: Quadrant) -> Self {
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

    fn get_force(&self, b: Body) -> Vec2 {
        let force = Vec2::new(0., 0.);
        // TODO: recursively walk the tree to add up forces 
        force
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
}