import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from typing import Any

class Tile:
    def __init__(self) -> None:
        
        self.id:int = None
        self.is_alive:bool = True

        self.rect: pygame.Rect = None
        self.mask: pygame.mask = None
        self.original_color = (50, 50, 50)
        self.color = self.original_color

        self.center_rect: pygame.Rect = None
        self.center_mask: pygame.mask = None
        self.mask_color = (255, 0, 0)

        self.sensors_centers = []
        self.sensors_masks = []     


        # Knowledge:
        self.neighbors:list[Tile] = []
        
        self.vector = np.array([0, 0], dtype=float)     # Shared
        self.vector_translation = np.array([0, 0], dtype=float)
        self.vector_rotation = np.array([0, 0], dtype=float)
        
        self.x: int = None                              # Shared
        self.y: int = None                              # Shared                  
        self.is_target:bool = False                     # Shared
        self.is_contact:bool = False                    # Shared
        
        self.object_center = np.array([0, 0], dtype=float)
        self.object_angle: float = 0
        
        self.target_center = np.array([0, 0], dtype=float)             
        self.target_angle: float = 0                 

        self.knowledge:dict[Tile, dict[str, Any]] = {}

        self.cluster = None
        
    def get_number_of_neighbors(self):
        if not self.is_alive:
            return None
        n = 0
        for neighbor in self.neighbors:
            if neighbor.is_alive:
                n += 1
        return n
    
    def get_dict(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'is_target': self.is_target,
            'is_contact': self.is_contact,
            'is_alive': self.is_alive,
            'number_of_neighbors': self.get_number_of_neighbors(),
            'object_center': self.object_center,
            'object_angle': self.object_angle,
            'target_center': self.target_center,
            'target_angle': self.target_angle,
            'vector_x': self.vector[0],
            'vector_y': self.vector[1],
            'vector_translation_x': self.vector_translation[0],
            'vector_translation_y': self.vector_translation[1],
            'vector_rotation_x': self.vector_rotation[0],
            'vector_rotation_y': self.vector_rotation[1],

        }
    def update_knowledge(self):
        if not self.is_alive:
            return
        for neighbor in self.neighbors:
            self.knowledge[neighbor] = {}
            self.knowledge[neighbor]['x'] = neighbor.x  #this is not going to change
            self.knowledge[neighbor]['y'] = neighbor.y  #this is not going to change

            self.knowledge[neighbor]['vector'] = neighbor.vector
            self.knowledge[neighbor]['vector_translation'] = neighbor.vector_translation
            self.knowledge[neighbor]['vector_rotation'] = neighbor.vector_rotation

            self.knowledge[neighbor]['is_target'] = neighbor.is_target
            self.knowledge[neighbor]['is_contact'] = neighbor.is_contact

            self.knowledge[neighbor]['object_center'] = neighbor.object_center
            self.knowledge[neighbor]['object_angle'] = neighbor.object_angle 

            self.knowledge[neighbor]['target_center'] = neighbor.target_center
            self.knowledge[neighbor]['target_angle'] = neighbor.target_angle

    def execute_behavior(self):
        if not self.is_alive:
            return
        
        if self.is_target:
            self.vector = np.array([0, 0], dtype=float)
            self.vector_translation = np.array([0, 0], dtype=float)
            self.vector_rotation = np.array([0, 0], dtype=float)
            return
        
        target_neighbors = [neighbor for neighbor in self.knowledge.keys() if self.knowledge[neighbor]['is_target']]
        if target_neighbors:
          
            #point to the target
            x = sum(self.knowledge[neighbor]['x'] for neighbor in target_neighbors) / len(target_neighbors)
            y = sum(self.knowledge[neighbor]['y'] for neighbor in target_neighbors) / len(target_neighbors)
            avg_target_position = np.array([x, y], dtype=float)
            my_position = np.array([self.x, self.y], dtype=float)
            vector_to_object = avg_target_position - my_position
            
            if np.linalg.norm(vector_to_object) != 0:
                self.vector_translation = vector_to_object / np.linalg.norm(vector_to_object)
                self.vector = self.vector_translation
            else:
                self.vector_translation = np.array([0, 0], dtype=float)
                self.vector = self.vector_translation
            
        else:
            x = sum(self.knowledge[neighbor]['vector_translation'][0] for neighbor in self.knowledge.keys()) / len(self.knowledge.keys())
            y = sum(self.knowledge[neighbor]['vector_translation'][1] for neighbor in self.knowledge.keys()) / len(self.knowledge.keys())
            
            vector_translation = np.array([x, y], dtype=float)
            
            if np.linalg.norm(vector_translation) != 0:
                self.vector_translation = vector_translation / np.linalg.norm(vector_translation)
                self.vector = self.vector_translation
            else:
                self.vector_translation = np.array([0, 0], dtype=float)
                self.vector = self.vector_translation
                        
            
        #Add information about the target to the knowledge
        avg_target_center = np.array([0, 0], dtype=float)
        avg_target_angle = 0
        
        for neighbor in self.knowledge.keys():
            avg_target_center += self.knowledge[neighbor]['target_center']
            avg_target_angle += self.knowledge[neighbor]['target_angle']
        
        avg_target_center = avg_target_center / len(self.knowledge)
        avg_target_angle = avg_target_angle / len(self.knowledge)
        
        self.target_center = avg_target_center
        self.target_angle = avg_target_angle
     
        if self.is_contact:
            object_center = np.array(self.object_center, dtype=float) + np.array([0.5, 0.5], dtype=float)
            tile_center = np.array([self.x, self.y], dtype=float) + np.array([0.5, 0.5], dtype=float)
            r =  tile_center - object_center
            
            perpendicular = np.array([-r[1], r[0]], dtype=float)
            if np.linalg.norm(perpendicular) != 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            else:
                perpendicular = np.array([0, 0], dtype=float)

            
            error = self.target_angle - self.object_angle
            #print(f'final: {self.target_angle}, initial: {self.object_angle}, error: {error}')
            if error > 180:
                error = error - 360
            self.vector_rotation =  2*(-error)/180*perpendicular
            
            self.vector = self.vector_translation + self.vector_rotation

        else:
            self.vector_rotation = np.array([0, 0], dtype=float)
            self.vector = self.vector_translation


    def die(self):
        self.is_alive = False
        self.color = (0,0,0)
        self.vector = np.array([0, 0], dtype=float)
        self.vector_translation = np.array([0, 0], dtype=float)
        self.vector_rotation = np.array([0, 0], dtype=float)

        for neighbor in self.knowledge.keys():
            neighbor.knowledge.pop(self)


    #GETTERS, SETTERS AND PRINTERS

    def set_as_target(self):
        self.is_target = True
        self.color = (255,0,0)

    def set_as_no_target(self):
        self.is_target = False
        self.color = self.original_color

    def set_as_contact(self):
        self.is_contact = True
        self.color = (255,160,122)
    
    def set_as_no_contact(self):
        self.is_contact = False
        self.color = self.original_color

    def get_coordinates(self):
        return (self.x, self.y)

    def print_details(self, show_knowledge:bool = False):
        print(f"Tile {self.id}")
        print(f"Position: ({self.x}, {self.y})")
        print(f"Vector: {self.vector}")
        print(f"Vector translation: {self.vector_translation}")
        print(f"Vector rotation: {self.vector_rotation}")
        print(f"Is target: {self.is_target}")
        print(f"Is contact: {self.is_contact}")
        print(f"Object angle: {self.object_angle}")
        print(f"Object center: {self.object_center}")

        print(f"Target angle: {self.target_angle}")
        print(f"Target center: {self.target_center}")
        print(f"Neighbors: {[neighbor.id for neighbor in self.neighbors]}")
        if show_knowledge:
            print(f"Knowledge:")#{self.knolwedge}")
            for key, value in self.knowledge.items():
                print(f"|    Tile {key.id}")
                for k, v in value.items():
                    print(f"|        | {k}: {v}")
        print()
    
    def __repr__(self) -> str:
        return f"Tile {self.id}"
    
    def __str__(self) -> str:
        return f"Tile {self.id}"

if __name__ == '__main__':
    pass
