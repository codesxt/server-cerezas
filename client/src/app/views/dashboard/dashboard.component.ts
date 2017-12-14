import { Component, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { environment } from '../../../environments/environment';

import { ImageService } from '../../services/image.service';

@Component({
  templateUrl: 'dashboard.component.html'
})
export class DashboardComponent {
  @ViewChild('fileInput') fileInput;
  baseURL: string = environment.apiUrl;

  results   : any = null;
  error     : any = null;

  constructor(
    private imageService : ImageService
  ) { }

  clearMessages(){
    this.results = null;
    this.error   = null;
  }

  uploadPhoto(){
    this.clearMessages();
    let fileBrowser = this.fileInput.nativeElement;
    if (fileBrowser.files && fileBrowser.files[0]) {
      const formData = new FormData();
      formData.append("image", fileBrowser.files[0]);
      this.imageService.uploadImage(formData)
      .subscribe(
        data => {
          console.log(data);
          this.results = data;
        },
        error => {
          console.log(error.json())
          this.error = error.json()
        }
      )
    }
  }

  imageChange(event){
    this.clearMessages();
    let inputFiles = event.target.files;
    if(inputFiles.length>0){
      //console.log(fileInput[0]);
      const formData = new FormData();
      formData.append("image", inputFiles[0]);
      this.imageService.uploadImage(formData)
      .subscribe(
        data => {
          console.log(data);
          this.results = data;
        },
        error => {
          console.log(error.json())
          this.error = error.json()
        }
      )
    }
  }
}
